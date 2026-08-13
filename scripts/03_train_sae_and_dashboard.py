import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from moe_interp.models import load_model, make_dirs
from moe_interp.data import PILE_DOMAINS, chunk_tokenize, load_pile_domains
from moe_interp.routing import load_expert_data
from moe_interp.sae import train_sae, get_feature_activations
from moe_interp.interp import build_dashboard_data
from moe_interp.dashboard import generate_interactive_dashboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="./moe_interp_out")
    parser.add_argument("--model", default="switch_base_8")
    parser.add_argument("--layer", required=True, help="e.g. enc_5")
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--d-hidden-mult", type=int, default=2)
    parser.add_argument("--l1-coef", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    dirs = make_dirs(args.base_dir)
    tokenizer, model, cfg = load_model(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_dir = os.path.join(dirs["cache"], f"layer_{args.layer}_meta")
    X_np, X_meta = load_expert_data(args.layer, args.expert, PILE_DOMAINS, dirs["cache"], meta_dir)
    print(f"Expert {args.expert} @ {args.layer}: {X_np.shape[0]} activations, dim={X_np.shape[1]}")

    sae, norm_stats = train_sae(
        X_np, d_hidden_mult=args.d_hidden_mult, l1_coef=args.l1_coef, epochs=args.epochs, device=device,
    )
    torch.save(
        {"state_dict": sae.state_dict(), "norm_stats": norm_stats, "d_in": sae.d_in, "d_hidden": sae.d_hidden},
        os.path.join(dirs["sae"], f"expert_{args.expert}_{args.layer}_sae.pt"),
    )

    F_acts = get_feature_activations(sae, X_np, norm_stats, device=device)
    density = (F_acts > 0).float().mean(dim=0)
    max_act = F_acts.max(dim=0).values
    alive_mask = density > 0
    print(f"{alive_mask.sum().item()}/{sae.d_hidden} features alive")

    domain_data = load_pile_domains(PILE_DOMAINS)
    _doc_token_cache = {}

    def get_doc_tokens(domain, doc_id):
        if domain not in _doc_token_cache:
            chunks = chunk_tokenize(domain_data[domain], tokenizer)
            _doc_token_cache[domain] = [tokenizer.convert_ids_to_tokens(ids[0]) for ids in chunks]
        return _doc_token_cache[domain][doc_id]

    candidate_feats = alive_mask.nonzero().squeeze(-1).tolist()
    dashboard_data = build_dashboard_data(
        candidate_feats, F_acts, X_meta, density, max_act,
        get_doc_tokens_fn=get_doc_tokens, k=20, window=8, min_examples=4,
    )

    out_path = os.path.join(dirs["figures"], f"feature_dashboard_expert{args.expert}_{args.layer}.html")
    generate_interactive_dashboard(dashboard_data, out_path, title=f"Expert {args.expert} @ {args.layer}")


if __name__ == "__main__":
    main()
