"""moe_interp — tools for studying token routing and learned features
inside sparse Mixture-of-Experts transformers (Switch Transformer family).

Pipeline stages, each in its own module:
    models   -- model registry + loading
    hooks    -- forward hooks that record routing decisions / expert inputs
    data     -- Pile-domain and multilingual Wikipedia sampling, tokenization
    routing  -- running the encoder and collecting per-token routing stats
    sae      -- sparse autoencoder for expert activations
    interp   -- feature examples, HTML dashboards, LLM autointerp export
    viz      -- matplotlib / HTML visualizations of routing behavior
"""

from .models import MODEL_CONFIGS, load_model, get_moe_layers
from .hooks import RoutingRecorder, ExpertActivationRecorder

__all__ = [
    "MODEL_CONFIGS",
    "load_model",
    "get_moe_layers",
    "RoutingRecorder",
    "ExpertActivationRecorder",
]
