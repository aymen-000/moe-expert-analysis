# Mixture of Experts: Are They Really Experts?

Interpretability for sparse Mixture-of-Experts transformers (Switch
Transformer family). This is the code behind a blog post on MoE routing
behavior and per-expert feature interpretability 

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Tested with `google/switch-base-8` (8 experts, 12 layers); 

## Pipeline

Each stage is a standalone script under `scripts/`, reading/writing to a
shared `--base-dir` (default `./moe_interp_out`):

```bash
BASE=./moe_interp_out
MODEL=switch_base_8

# 1. Routing-by-domain stats + token-level routing visualization
python scripts/01_collect_domain_routing.py --base-dir $BASE --model $MODEL

# 2. Cache raw activations flowing into each expert, for SAE training
python scripts/02_collect_expert_activations.py --base-dir $BASE --model $MODEL \
    --layers enc_1 enc_5 enc_11

# 3. Train an SAE on one expert and build its feature dashboard
python scripts/03_train_sae_and_dashboard.py --base-dir $BASE --model $MODEL \
    --layer enc_5 --expert 3

# 4. Full autointerp export: SAE + top features + dashboards + summary figure,
#    for every expert in the given layers
python scripts/04_autointerp_export.py --base-dir $BASE --model $MODEL \
    --layers enc_1 enc_11

# 5. (optional) Label exported features with an LLM
export GEMINI_API_KEY=...
python scripts/05_label_features_with_llm.py --base-dir $BASE

# 6. Cross-lingual routing comparison
python scripts/06_language_routing.py --base-dir $BASE --model $MODEL --layer enc_5
```


## Notes

- Everything here targets Switch Transformer's top-1 softmax router; the
  hook logic in `hooks.py` also has branches for DeepSeek- and
  Mixtral-style gating if you adapt it to another architecture — you'll
  need to add a matching entry to `MODEL_CONFIGS` in `models.py`.
- The domain-routing sample (`NeelNanda/pile-10k`) and the multilingual
  sample (`wikimedia/wikipedia`) are both pulled via 🤗 `datasets` at
  runtime , no manual downloads needed.
- SAE hyperparameters (`d_hidden_mult`, `l1_coef`, epochs) are set for a
  base-8 model on a single GPU/CPU run; expect to retune for larger models.
