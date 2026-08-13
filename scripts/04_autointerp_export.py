import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from moe_interp.models import load_model, make_dirs
from moe_interp.data import PILE_DOMAINS, chunk_tokenize, load_pile_domains
from moe_interp.autointerp import run_autointerp_export, build_all_prompts, load_top_feature_records
from moe_interp.dashboard import build_dashboard_payload, generate_expert_feature_dashboard
from moe_interp.viz import discover_autointerp_files, load_layer_features, plot_top_features_per_expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="./moe_interp_out")
    parser.add_argument("--model", default="switch_base_8")
    parser.add_argument("--layers", nargs="+", required=True, help="e.g. enc_0 enc_11")
    parser.add_argument("--top-features-per-expert", type=int, default=40)
    parser.add_argument("--examples-per-feature", type=int, default=200)
    parser.add_argument("--min-tokens-for-sae", type=int, default=3000)
    parser.add_argument("--top-k-in-figure", type=int, default=6)
    args = parser.parse_args()

    dirs = make_dirs(args.base_dir)
    tokenizer, model, cfg = load_model(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    domain_data = load_pile_domains(PILE_DOMAINS)
    _doc_token_cache = {}

    def get_doc_tokens(domain, doc_id):
        if domain not in _doc_token_cache:
            chunks = chunk_tokenize(domain_data[domain], tokenizer)
            _doc_token_cache[domain] = [tokenizer.convert_ids_to_tokens(ids[0]) for ids in chunks]
        return _doc_token_cache[domain][doc_id]

    autointerp_index = run_autointerp_export(
        model, tokenizer, cfg, PILE_DOMAINS, args.layers,
        dirs["cache"], dirs["sae"], dirs["autointerp"], get_doc_tokens,
        top_features_per_expert=args.top_features_per_expert,
        examples_per_feature=args.examples_per_feature,
        min_tokens_for_sae=args.min_tokens_for_sae,
        device=device,
    )
    build_all_prompts(autointerp_index, dirs["autointerp"])

    available = discover_autointerp_files(dirs["autointerp"])
    n_experts = cfg["n_routed_experts"]

    for layer_key in args.layers:
        feats, skipped = load_layer_features(available, layer_key, n_experts)
        print(f"{layer_key}: experts plotted = {sorted(feats.keys())}, skipped = {skipped}")
        plot_top_features_per_expert(
            feats, layer_key, n_experts, dirs["figures"],
            f"top_features_per_expert_{layer_key}", top_k=args.top_k_in_figure,
        )

        top_records = load_top_feature_records(available, layer_key, top_k=args.top_k_in_figure)
        payload = build_dashboard_payload(top_records, layer_key)
        out_path = os.path.join(dirs["figures"], f"expert_feature_dashboard_{layer_key}.html")
        generate_expert_feature_dashboard(payload, out_path, title=f"Top Features per Expert — {layer_key}")


if __name__ == "__main__":
    main()
