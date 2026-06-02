"""
EduPredict Math — Unified Inference Engine & REST API
======================================================
Provides a unified API to switch between:
1. LSTM (Multivariate Response Time DKT)
2. Causal Cross-Transformer (SAINT/SAKT style DKT)

Handles input student interaction sequence, processes it according to the selected model,
aggregates mastery scores by top categories, and triggers personalized Generative AI 
hints using Google Gemini based on student preference.
"""

import os
import json
import math
import logging
from typing import Optional, List, Dict, Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DKTInferenceEngine")

# ──────────────────────────────────────────────────────────────
# Global Configuration
# ──────────────────────────────────────────────────────────────
# Easy Switch Model Choice: 'LSTM' or 'Transformer'
MODEL_TYPE = "Transformer"

# Thresholds
STRUGGLE_THRESHOLD = 0.50     # Skills below this are flagged as struggling
LENGTH_THRESHOLD = 3          # strictly greater than 3 to trigger GenAI
AVG_LEN_THRESHOLD = 1         # Min interactions for skill to be included in top category avg

# Model directories and path mappings
if MODEL_TYPE == "LSTM":
    MODEL_PATH = "final/lstm_dkt_model_train/lstm_dkt_model.keras"
    VOCAB_PATH = "final/lstm_dkt_model_train/vocab.json"
    MAX_SEQ_LEN = 100
elif MODEL_TYPE == "Transformer":
    MODEL_PATH = "final/causal_cross_transformer_dkt_model_train/causal_cross_transformer_dkt_model.keras"
    VOCAB_PATH = "final/causal_cross_transformer_dkt_model_train/vocab.json"
    MAX_SEQ_LEN = 100
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

TOP_CATEGORY_PATH = "top_category.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ──────────────────────────────────────────────────────────────
# Keras Custom Layers Definition
# ──────────────────────────────────────────────────────────────

class TemporalAttentionLayer(tf.keras.layers.Layer):
    """LSTM-specific custom temporal attention layer."""
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W_query = tf.keras.layers.Dense(self.units, name="attn_query")
        self.W_key = tf.keras.layers.Dense(self.units, name="attn_key")
        self.W_value = tf.keras.layers.Dense(self.units, name="attn_value")
        super().build(input_shape)

    def call(self, lstm_outputs, mask=None):
        Q = self.W_query(lstm_outputs)
        K = self.W_key(lstm_outputs)
        V = self.W_value(lstm_outputs)

        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)

        seq_len = tf.shape(scores)[1]
        causal = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)
        scores = scores + (1.0 - causal) * (-1e9)

        if mask is not None:
            pad_mask = tf.cast(mask[:, tf.newaxis, :], tf.float32)
            scores = scores + (1.0 - pad_mask) * (-1e9)

        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, V)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class MultivariateResponseTimeDKT(tf.keras.Model):
    """LSTM DKT Model wrapper."""
    def __init__(self, vocab_size: int, embed_dim: int = 64, lstm_units: int = 64,
                 attention_units: int = 64, dropout_rate: float = 0.4, l2_reg: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.proj_inputs = tf.keras.layers.Dense(embed_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.proj_hist = tf.keras.layers.Dense(embed_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.spatial_dropout = tf.keras.layers.SpatialDropout1D(dropout_rate)
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, use_cudnn=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name="lstm")
        self.attention = TemporalAttentionLayer(attention_units)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout_out = tf.keras.layers.Dropout(dropout_rate)
        self.output_dense = tf.keras.layers.Dense(vocab_size, activation="sigmoid", name="skill_output")

    def call(self, inputs, training=False):
        mask = tf.reduce_any(tf.not_equal(inputs, 0.0), axis=-1)
        Q = inputs[:, :, :self.vocab_size]
        C = inputs[:, :, self.vocab_size : 2 * self.vocab_size]
        R = inputs[:, :, 2 * self.vocab_size :]

        cum_attempts = tf.cumsum(Q, axis=1)
        cum_corrects = tf.cumsum(C, axis=1)
        skill_specific_acc = cum_corrects / tf.maximum(cum_attempts, 1.0)
        hist_features = tf.concat([cum_attempts, cum_corrects, skill_specific_acc, R], axis=-1)

        base_interaction = tf.concat([Q, C], axis=-1)
        x_emb = self.proj_inputs(base_interaction)
        h_emb = self.proj_hist(hist_features)

        x = tf.concat([x_emb, h_emb], axis=-1)
        x = self.spatial_dropout(x, training=training)

        lstm_out = self.lstm(x, mask=mask)
        context = self.attention(lstm_out, mask=mask)

        combined = tf.concat([lstm_out, context], axis=-1)
        combined = self.layer_norm(combined)
        combined = self.dropout_out(combined, training=training)
        return self.output_dense(combined)

    def get_config(self):
        return {"vocab_size": self.vocab_size}


class MultiHeadTemporalAttention(tf.keras.layers.Layer):
    """Transformer-specific Multi-Head Causal Attention."""
    def __init__(self, units: int, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        assert units % num_heads == 0, "units must be divisible by num_heads"
        self.units = units
        self.num_heads = num_heads
        self.head_dim = units // num_heads

    def build(self, input_shape):
        self.W_query = tf.keras.layers.Dense(self.units, name="mha_query")
        self.W_key = tf.keras.layers.Dense(self.units, name="mha_key")
        self.W_value = tf.keras.layers.Dense(self.units, name="mha_value")
        self.W_out = tf.keras.layers.Dense(self.units, name="mha_output")
        super().build(input_shape)

    def _split_heads(self, x):
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch, seq_len, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key_value, mask=None):
        Q = self._split_heads(self.W_query(query))
        K = self._split_heads(self.W_key(key_value))
        V = self._split_heads(self.W_value(key_value))
        d_k = tf.cast(self.head_dim, tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
        seq_len = tf.shape(scores)[2]
        causal = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)
        scores = scores + (1.0 - causal[tf.newaxis, tf.newaxis, :, :]) * (-1e9)
        if mask is not None:
            pad_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)
            scores = scores + (1.0 - pad_mask) * (-1e9)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, V)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        batch = tf.shape(context)[0]
        seq_len_out = tf.shape(context)[1]
        context = tf.reshape(context, (batch, seq_len_out, self.units))
        return self.W_out(context)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "num_heads": self.num_heads})
        return config


class TransformerDecoderBlock(tf.keras.layers.Layer):
    """Transformer decoder block."""
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.att = MultiHeadTemporalAttention(self.embed_dim, self.num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ffn_dim, activation="gelu", name="ffn_dense_1"),
            tf.keras.layers.Dense(self.embed_dim, name="ffn_dense_2")
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm_1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm_2")
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, query, key_value, training=False, mask=None):
        attn_output = self.att(query, key_value, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(query + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


class TransformerMultivariateDKT(tf.keras.Model):
    """Causal Cross-Transformer DKT model."""
    def __init__(self, vocab_size, embed_dim=64, attention_units=64,
                 num_transformer_blocks=2, num_attn_heads=4, 
                 ffn_dim=128, dropout_rate=0.4, l2_reg=1e-4,
                 max_seq_len=100, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self._config = dict(embed_dim=embed_dim, attention_units=attention_units,
                             num_transformer_blocks=num_transformer_blocks,
                             num_attn_heads=num_attn_heads, ffn_dim=ffn_dim,
                             dropout_rate=dropout_rate, l2_reg=l2_reg)

        reg = tf.keras.regularizers.l2(l2_reg)
        self.proj_target = tf.keras.layers.Dense(attention_units, activation="gelu", kernel_regularizer=reg)
        self.proj_inputs = tf.keras.layers.Dense(embed_dim, activation="gelu", kernel_regularizer=reg)
        self.proj_hist = tf.keras.layers.Dense(embed_dim, activation="gelu", kernel_regularizer=reg)
        self.combined_proj = tf.keras.layers.Dense(attention_units, activation="gelu", kernel_regularizer=reg)
        self.pos_embedding = tf.keras.layers.Embedding(max_seq_len, attention_units, name="pos_emb")
        self.input_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.transformer_blocks = [
            TransformerDecoderBlock(attention_units, num_attn_heads, ffn_dim, dropout_rate, name=f"transformer_block_{i}")
            for i in range(num_transformer_blocks)
        ]
        self.output_dense = tf.keras.layers.Dense(1, activation="sigmoid", name="skill_output")

    def call(self, inputs, training=False):
        Q_next = inputs[:, :, :self.vocab_size]
        Q_past = inputs[:, :, self.vocab_size : 2 * self.vocab_size]
        C_past = inputs[:, :, 2 * self.vocab_size : 3 * self.vocab_size]
        R_past = inputs[:, :, 3 * self.vocab_size :]
        
        mask = tf.reduce_any(tf.not_equal(Q_next, 0.0), axis=-1)
        
        cum_attempts = tf.cumsum(Q_past, axis=1)
        cum_corrects = tf.cumsum(C_past, axis=1)
        skill_specific_acc = cum_corrects / tf.maximum(cum_attempts, 1.0)
        
        overall_attempts = tf.cumsum(tf.reduce_sum(Q_past, axis=-1, keepdims=True), axis=1)
        overall_corrects = tf.cumsum(tf.reduce_sum(C_past, axis=-1, keepdims=True), axis=1)
        overall_acc = overall_corrects / tf.maximum(overall_attempts, 1.0)
        
        hist_features = tf.concat([
            cum_attempts, cum_corrects, skill_specific_acc,
            overall_attempts, overall_corrects, overall_acc, R_past
        ], axis=-1)
        
        target_emb = self.proj_target(Q_next)
        x_past_emb = self.proj_inputs(tf.concat([Q_past, C_past], axis=-1))
        h_past_emb = self.proj_hist(hist_features)
        past_emb = tf.concat([x_past_emb, h_past_emb], axis=-1)
        past_emb = self.combined_proj(past_emb)
        
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(seq_len)
        pos_emb = self.pos_embedding(positions)[tf.newaxis, :, :]
        
        target_emb = target_emb + pos_emb
        past_emb = past_emb + pos_emb
        
        target_emb = self.input_dropout(target_emb, training=training)
        past_emb = self.input_dropout(past_emb, training=training)
        
        x = target_emb
        for block in self.transformer_blocks:
            x = block(query=x, key_value=past_emb, training=training, mask=mask)
        return self.output_dense(x)

    def get_config(self):
        return {"vocab_size": self.vocab_size, **self._config}

# ──────────────────────────────────────────────────────────────
# Global Variables (Populated at Startup)
# ──────────────────────────────────────────────────────────────
_model = None
_vocab = None
_vocab_size = None
_idx_to_name = None
_skill_id_to_name = None
_top_categories = None

# ──────────────────────────────────────────────────────────────
# Artifact Ingestion & Model Loading
# ──────────────────────────────────────────────────────────────

def load_artefacts():
    """Load the trained model and vocabulary based on global config."""
    logger.info("=" * 60)
    logger.info(f"LOADING ARTIFACTS FOR MODEL_TYPE: {MODEL_TYPE}")
    logger.info(f"Model File: {MODEL_PATH}")
    logger.info(f"Vocab File: {VOCAB_PATH}")
    logger.info("=" * 60)
    
    custom_objects = {
        "TemporalAttentionLayer": TemporalAttentionLayer,
        "MultivariateResponseTimeDKT": MultivariateResponseTimeDKT,
        "MultiHeadTemporalAttention": MultiHeadTemporalAttention,
        "TransformerDecoderBlock": TransformerDecoderBlock,
        "TransformerMultivariateDKT": TransformerMultivariateDKT,
    }
    
    if not os.path.exists(VOCAB_PATH):
        logger.error(f"Vocabulary file not found at {VOCAB_PATH}!")
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")
        
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    vocab = meta["vocab"]
    vocab_size = meta["vocab_size"]
    idx_to_name = {int(k): v for k, v in meta["idx_to_name"].items()}
    skill_id_to_name = meta.get("skill_id_to_name", {})
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}!")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
    # Load model with compile=False to avoid needing custom loss
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False
    )
    
    logger.info(f"Loaded vocab of size {vocab_size} and idx_to_name of size {len(idx_to_name)}.")
    logger.info("Model loaded successfully with compile=False.")
    return model, vocab, vocab_size, idx_to_name, skill_id_to_name


def load_top_categories():
    """Load the categories from top_category.json."""
    logger.info(f"Loading top categories from: {TOP_CATEGORY_PATH}")
    if not os.path.exists(TOP_CATEGORY_PATH):
        logger.error(f"top_category.json not found at {TOP_CATEGORY_PATH}!")
        raise FileNotFoundError(f"top_category.json not found: {TOP_CATEGORY_PATH}")
        
    with open(TOP_CATEGORY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter out "Miscellaneous / Other"
    filtered = [cat for cat in data if cat["title"] != "Miscellaneous / Other"]
    logger.info(f"Loaded {len(filtered)} active top categories.")
    return filtered

# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────

def encode_multi_hot(skill_id_str: str, vocab: dict, vocab_size: int) -> np.ndarray:
    """Encode compound skill string (e.g. '10_12') into a multi-hot vector."""
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ind in str(skill_id_str).split("_"):
        vec[vocab.get(ind, 0)] = 1.0
    return vec


def predict_mastery(
    student_history: list[tuple[str, int, int]],
    model: tf.keras.Model,
    vocab: dict,
    vocab_size: int,
    idx_to_name: dict[int, str],
) -> np.ndarray:
    """
    Run DKT model forward pass to output predicted mastery probabilities
    for all vocab_size skills.
    
    student_history list elements: (skill_id, correctness, ms_first_response)
    """
    T = len(student_history)
    if T < 1:
        logger.info("Empty history. Returning neutral default mastery of 0.5.")
        return np.full(vocab_size, 0.5, dtype=np.float32)

    logger.info(f"[Inference Engine] Input Sequence Length T = {T} (MODEL_TYPE: {MODEL_TYPE})")
    
    # Pre-encode all historical interactions
    Q = [encode_multi_hot(skill_id, vocab, vocab_size) for skill_id, _, _ in student_history]
    C = [Q[t] * float(correct) for t, (_, correct, _) in enumerate(student_history)]
    
    # Process log response times (capped at 8 or 10 min, convert to log(1 + seconds))
    R = []
    for _, _, ms in student_history:
        ms_val = float(ms)
        if ms_val < 0: ms_val = 0.0
        # Capping at 8 min (480000ms) for LSTM and 10 min (600000ms) for Transformer
        cap_val = 480000.0 if MODEL_TYPE == "LSTM" else 600000.0
        ms_val = min(ms_val, cap_val)
        r_val = math.log1p(ms_val / 1000.0)
        R.append(np.array([r_val], dtype=np.float32))

    if MODEL_TYPE == "LSTM":
        # LSTM input shape: (B, T, 2 * vocab_size + 1)
        # Create input array of shape (1, MAX_SEQ_LEN, vocab_size * 2 + 1)
        X = np.zeros((1, MAX_SEQ_LEN, vocab_size * 2 + 1), dtype=np.float32)
        
        # Pre-pad: active steps are placed at the end of the sequence
        n_steps = min(T, MAX_SEQ_LEN)
        pad_offset = MAX_SEQ_LEN - n_steps
        
        for t in range(n_steps):
            hist_t = T - n_steps + t
            x_step = np.concatenate([Q[hist_t], C[hist_t], R[hist_t]])
            X[0, pad_offset + t, :] = x_step
            
        logger.info(f"[LSTM Prep] Formatted sequence of length {n_steps} (pre-padded by {pad_offset} steps). Shape: {X.shape}")
        
        # Forward pass
        predictions = model(X, training=False)  # Shape: (1, MAX_SEQ_LEN, vocab_size)
        
        # Extract predictions at the final step index (MAX_SEQ_LEN - 1)
        probs = predictions[0, MAX_SEQ_LEN - 1, :].numpy()
        
    elif MODEL_TYPE == "Transformer":
        # Transformer input shape: (B, T, 3 * vocab_size + 1)
        # We need target one-hot skill in inputs[:, :, :vocab_size].
        # To get the predicted mastery of ALL vocab_size skills, we construct a batch of size vocab_size
        X = np.zeros((vocab_size, T, vocab_size * 3 + 1), dtype=np.float32)
        
        # Fill steps 0 ... T-2 (historical steps)
        for t in range(T - 1):
            # Target skill for step t is the actual skill at t+1
            x_step = np.concatenate([Q[t + 1], Q[t], C[t], R[t]])
            X[:, t, :] = x_step
            
        # Fill the final step T-1 for each candidate target skill index
        for i in range(vocab_size):
            Q_target = np.zeros(vocab_size, dtype=np.float32)
            Q_target[i] = 1.0
            x_last = np.concatenate([Q_target, Q[T - 1], C[T - 1], R[T - 1]])
            X[i, T - 1, :] = x_last
            
        # Truncate if T exceeds MAX_SEQ_LEN
        n_steps = T
        if n_steps > MAX_SEQ_LEN:
            X = X[:, -MAX_SEQ_LEN:, :]
            n_steps = MAX_SEQ_LEN
            
        # Post-pad: pad with zeros at the end if n_steps < MAX_SEQ_LEN
        if n_steps < MAX_SEQ_LEN:
            pad = MAX_SEQ_LEN - n_steps
            X_padded = np.zeros((vocab_size, MAX_SEQ_LEN, vocab_size * 3 + 1), dtype=np.float32)
            X_padded[:, :n_steps, :] = X
            X = X_padded
            
        logger.info(f"[Transformer Prep] Formatted batch of size {vocab_size} with seq length {n_steps}. Shape: {X.shape}")
        
        # Forward pass
        y_pred = model(X, training=False)  # Shape: (vocab_size, MAX_SEQ_LEN, 1)
        
        # Extract probabilities at the last valid time-step
        valid_step_idx = n_steps - 1
        probs = y_pred[:, valid_step_idx, 0].numpy()  # Shape: (vocab_size,)
        
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
        
    # Log raw probabilities sample
    logger.info(f"[Model Output Log] Raw prediction probs (first 10 skills): {probs[:10]}")
    return probs

# ──────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────

class Interaction(BaseModel):
    skill_id: str = Field(..., description="Skill ID or underscore-separated ID compound like '2_37'")
    correctness: int = Field(..., description="Is the response correct (1) or incorrect (0)")
    ms_first_response: int = Field(..., description="Time taken to answer in milliseconds")
    question: Optional[str] = Field(None, description="The text of the question, if available")


class PredictRequest(BaseModel):
    student_history: List[Interaction] = Field(..., description="Sequence of past student interactions")
    personal_preference: str = Field(..., description="Generative AI prompt style personalization preference")


class PredictResponse(BaseModel):
    category_mastery: Dict[str, Optional[float]] = Field(..., description="Mastery scores of the 6 top categories")
    explanation: Optional[str] = Field(None, description="Supportive explanation generated by Gemini AI (if triggered)")

# ──────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="EduPredict Math — Unified Inference API",
    description="Serving LSTM & Cross-Transformer DKT models with personalized Gemini support.",
    version="1.0.0"
)


@app.on_event("startup")
def startup_event():
    global _model, _vocab, _vocab_size, _idx_to_name, _skill_id_to_name, _top_categories
    _model, _vocab, _vocab_size, _idx_to_name, _skill_id_to_name = load_artefacts()
    _top_categories = load_top_categories()
    logger.info("Application successfully initialized and ready for requests.")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info("=" * 80)
    logger.info("INGESTED NEW PREDICT REQUEST")
    logger.info("=" * 80)
    
    # ── Step 1: Extract History ──
    history = [(item.skill_id, item.correctness, item.ms_first_response) for item in req.student_history]
    logger.info(f"Student history loaded with {len(history)} total interactions.")
    
    if len(history) == 0:
        raise HTTPException(status_code=400, detail="Student history cannot be empty.")
        
    # ── Step 2: Run DKT Model Inference ──
    probs = predict_mastery(history, _model, _vocab, _vocab_size, _idx_to_name)
    
    # Track per-skill historical interaction count
    # Let's count how many times each vocabulary index was active in the history
    skill_history_lens = {}
    for item in req.student_history:
        active_indices = [_vocab.get(s, 0) for s in str(item.skill_id).split("_")]
        for idx in active_indices:
            skill_history_lens[idx] = skill_history_lens.get(idx, 0) + 1
            
    logger.info(f"Historical interaction counts mapped for {len(skill_history_lens)} unique skill indices.")
    
    # ── Step 3: Check GenAI Trigger for Most Recent Question ──
    most_recent = req.student_history[-1]
    recent_skill_ids = str(most_recent.skill_id).split("_")
    logger.info(f"Most recent question skills to check: {recent_skill_ids}")
    
    struggling_skills_triggered = []
    
    for sid in recent_skill_ids:
        idx = _vocab.get(sid, 0)
        mastery = float(probs[idx])
        history_len = skill_history_lens.get(idx, 0)
        skill_name = _idx_to_name.get(idx, f"Skill_{idx}")
        
        logger.info(f"Checking trigger -> Skill '{sid}' ({skill_name}): mastery = {mastery:.4f}, history sequence len = {history_len}")
        
        if mastery < STRUGGLE_THRESHOLD:
            if history_len > LENGTH_THRESHOLD:
                logger.info(f"-> TRIGGER MET! mastery {mastery:.4f} < {STRUGGLE_THRESHOLD} AND history {history_len} > {LENGTH_THRESHOLD}")
                struggling_skills_triggered.append(skill_name)
            else:
                logger.info(f"-> Mastery is below threshold but history len ({history_len}) is not > threshold ({LENGTH_THRESHOLD}).")
                
    # Call Gemini GenAI if triggered
    explanation = None
    if struggling_skills_triggered:
        explanation = trigger_gemini_explanation(
            skills=struggling_skills_triggered,
            preference=req.personal_preference,
            question=most_recent.question
        )
    else:
        logger.info("Generative AI was not triggered (no skills met both mastery and history length triggers).")
        
    # ── Step 4: Top Category Mastery Score Aggregation ──
    category_mastery = {}
    
    for cat in _top_categories:
        title = cat["title"]
        included_str_indices = cat["skill_id_included"]
        
        cat_scores = []
        
        for idx_str in included_str_indices:
            idx = int(idx_str)
            # Ensure it is within our vocabulary
            if idx in _idx_to_name:
                history_len = skill_history_lens.get(idx, 0)
                if history_len >= AVG_LEN_THRESHOLD:
                    mastery_val = float(probs[idx])
                    cat_scores.append(mastery_val)
                    
        if len(cat_scores) > 0:
            avg_val = sum(cat_scores) / len(cat_scores)
            category_mastery[title] = round(avg_val, 4)
            logger.info(f"Category '{title}': Avg of {len(cat_scores)} eligible skills = {category_mastery[title]:.4f}")
        else:
            category_mastery[title] = None
            logger.info(f"Category '{title}': No skills met the avg_len_threshold of {AVG_LEN_THRESHOLD}. Returning null.")
            
    # Return consolidated response
    logger.info("Successfully completed predict request.")
    logger.info(f"Final output categories: {category_mastery}")
    return PredictResponse(category_mastery=category_mastery, explanation=explanation)


def trigger_gemini_explanation(skills: List[str], preference: str, question: Optional[str] = None) -> str:
    """Call Google Gemini Generative AI to generate personalized hint using their hobby/interest."""
    logger.info(f"[GenAI Trigger] Contacting Gemini for struggling skills: {skills}")
    logger.info(f"[GenAI Trigger] Student Hobby: \"{preference}\"")
    if question:
        logger.info(f"[GenAI Trigger] Specific Question: \"{question}\"")
    
    prompt = (
        f"A student is struggling with the following math concepts: {', '.join(skills)}.\n"
    )
    if question:
        prompt += f"The specific math question they are struggling with is: \"{question}\"\n"
        
    prompt += (
        f"The student's hobby / interest is: \"{preference}\"\n\n"
        "Please generate a warm, encouraging math hint in Bahasa Indonesia. "
        "The response MUST focus heavily on solving this specific question step-by-step "
        f"using their hobby (\"{preference}\") as a creative analogy, theme, or context to make it highly engaging and easy to understand. "
        "If the question requires step-by-step mathematical problem solving, show the clear step-by-step calculations. "
        "Keep the overall response extremely concise and strictly under 10 lines total."
    )
    
    logger.info(f"[Gemini Prompt]\n{prompt}")
    
    if not GEMINI_API_KEY:
        placeholder = (
            f"[Gemini API key not configured] "
            f"Keep practicing! Revisit the core rules of {', '.join(skills)}. "
            f"Try taking it step by step — you are fully capable of masteries like this!"
        )
        logger.warning("GEMINI_API_KEY is not set. Returning offline placeholder.")
        return placeholder
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()
        logger.info(f"[Gemini Response]\n{text}")
        return text
    except Exception as e:
        logger.error(f"Error calling Gemini: {e}")
        fallback = (
            f"[Gemini API connection error] "
            f"Review the key rules for {', '.join(skills)}. "
            f"Draw a picture or break it down into simple equations — you've got this!"
        )
        return fallback

# ──────────────────────────────────────────────────────────────
# Standalone Smoke Test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Let's perform a standalone smoke-test
    logger.info("=" * 60)
    logger.info("RUNNING STANDALONE SMOKE-TEST")
    logger.info("=" * 60)
    
    # Initialize
    _model, _vocab, _vocab_size, _idx_to_name, _skill_id_to_name = load_artefacts()
    _top_categories = load_top_categories()
    
    # Create a mock student history with compound skills
    mock_history = [
        ("2", 1, 12000),      # Circle Graph (idx 2)
        ("2_37", 0, 25000),   # Circle Graph & Addition Whole Numbers (idx 2, idx 25)
        ("2", 0, 18000),      # Circle Graph (idx 2)
        ("2", 0, 15000),      # Circle Graph (idx 2) -> Length for idx 2 is 4 (which is > 3 trigger threshold)
    ]
    
    logger.info("\n--- STEP 1: Running Predict Mastery ---")
    probs = predict_mastery(mock_history, _model, _vocab, _vocab_size, _idx_to_name)
    
    # Track per-skill historical length
    mock_lens = {}
    for item in mock_history:
        indices = [_vocab.get(s, 0) for s in str(item[0]).split("_")]
        for idx in indices:
            mock_lens[idx] = mock_lens.get(idx, 0) + 1
            
    logger.info("\n--- STEP 2: Running GenAI Trigger Check on Last Interaction ---")
    most_recent_skill = mock_history[-1][0]
    for s in most_recent_skill.split("_"):
        idx = _vocab.get(s, 0)
        mastery = float(probs[idx])
        hlen = mock_lens.get(idx, 0)
        logger.info(f"Skill '{s}' (name: {_idx_to_name[idx]}): mastery = {mastery:.4f}, history count = {hlen}")
        
    logger.info("\n--- STEP 3: Running Category Aggregation ---")
    for cat in _top_categories:
        title = cat["title"]
        scores = []
        for sidx_str in cat["skill_id_included"]:
            sidx = int(sidx_str)
            if sidx in _idx_to_name and mock_lens.get(sidx, 0) >= AVG_LEN_THRESHOLD:
                scores.append(float(probs[sidx]))
        avg = sum(scores) / len(scores) if scores else None
        logger.info(f"  Category '{title}': avg = {avg}")
        
    logger.info("\nStandalone Smoke-Test finished successfully!")
