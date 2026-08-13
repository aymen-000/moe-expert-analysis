import os
import json
import time

from .routing import load_expert_data
from .sae import train_sae, get_feature_activations, rank_features_by_importance
from .interp import token_routing_profile, collect_examples_for_llm, build_llm_prompt


def run_autointerp_export(model, tokenizer, cfg, domains, layers_to_study, cache_dir, sae_dir,
                           autointerp_dir, get_doc_tokens_fn,
                           top_features_per_expert=40, examples_per_feature=200,
                           min_tokens_for_sae=3000, d_hidden_mult=3, l1_coef=1e-4, epochs=100,
                           device="cuda"):
    autointerp_index = []

    for layer_key in layers_to_study:
        print(f"\n{'#'*70}\nLAYER {layer_key}\n{'#'*70}")
        meta_dir = os.path.join(cache_dir, f"layer_{layer_key}_meta")

        for expert_idx in range(cfg["n_routed_experts"]):
            print(f"\n--- expert {expert_idx} ---")
            domain_counts, _ = token_routing_profile(expert_idx, meta_dir, domains)
            if domain_counts is None:
                print("no data, skipping")
                continue
            n_tokens = int(domain_counts.sum())
            if n_tokens < min_tokens_for_sae:
                print(f"only {n_tokens} tokens, skipping SAE")
                continue

            X_np_e, X_meta_e = load_expert_data(layer_key, expert_idx, domains, cache_dir, meta_dir)

            sae_e, norm_stats_e = train_sae(
                X_np_e, d_hidden_mult=d_hidden_mult, l1_coef=l1_coef, epochs=epochs, device=device,
            )
            import torch
            torch.save(
                {"state_dict": sae_e.state_dict(), "norm_stats": norm_stats_e,
                 "d_in": sae_e.d_in, "d_hidden": sae_e.d_hidden},
                os.path.join(sae_dir, f"expert_{expert_idx}_{layer_key}_sae.pt"),
            )

            F_acts_e = get_feature_activations(sae_e, X_np_e, norm_stats_e, device=device)
            top_feats = rank_features_by_importance(F_acts_e, top_n=top_features_per_expert)
            print(f"{len(top_feats)} candidate features ranked by activation mass")

            feature_records = []
            for feat_idx in top_feats:
                examples = collect_examples_for_llm(
                    feat_idx, F_acts_e, X_meta_e, get_doc_tokens_fn,
                    n_examples=examples_per_feature, window=10,
                )
                if len(examples) < 5:  
                    continue
                density_val = (F_acts_e[:, feat_idx] > 0).float().mean().item()
                feature_records.append({
                    "feature_id": feat_idx,
                    "layer": layer_key,
                    "expert_id": expert_idx,
                    "density": round(density_val, 5),
                    "n_examples": len(examples),
                    "examples": examples,
                })

            out_path = os.path.join(autointerp_dir, f"{layer_key}_expert{expert_idx}_features.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "layer": layer_key,
                    "expert_id": expert_idx,
                    "n_tokens_total": n_tokens,
                    "n_features_exported": len(feature_records),
                    "features": feature_records,
                }, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(feature_records)} features -> {out_path}")
            autointerp_index.append({
                "layer": layer_key, "expert_id": expert_idx,
                "n_features_exported": len(feature_records), "path": out_path,
            })

    with open(os.path.join(autointerp_dir, "manifest.json"), "w") as f:
        json.dump(autointerp_index, f, indent=2)
    print(f"\nDone. {len(autointerp_index)} expert files written to {autointerp_dir}")
    return autointerp_index


def build_all_prompts(autointerp_index, autointerp_dir):
    """Builds LLM autointerp prompts for every exported feature and writes
    them to {autointerp_dir}/all_prompts.jsonl.
    """
    all_prompts = []
    for entry in autointerp_index:
        with open(entry["path"]) as f:
            data = json.load(f)
        for feat in data["features"]:
            prompt = build_llm_prompt(feat)
            all_prompts.append({
                "layer": feat["layer"], "expert_id": feat["expert_id"],
                "feature_id": feat["feature_id"], "prompt": prompt,
            })

    prompts_path = os.path.join(autointerp_dir, "all_prompts.jsonl")
    with open(prompts_path, "w", encoding="utf-8") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"{len(all_prompts)} prompts written to {prompts_path}")
    return prompts_path


def label_features_with_llm(prompts_path, out_path, model_id="gemini-2.5-flash",
                             max_features=None, sleep_s=0.5):
    from google import genai
    client = genai.Client()  

    with open(prompts_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if max_features is not None:
        records = records[:max_features]

    labeled = []
    for i, rec in enumerate(records):
        try:
            resp = client.models.generate_content(model=model_id, contents=rec["prompt"])
            label = (resp.text or "").strip()
        except Exception as e:
            label = f"[ERROR: {e}]"

        rec_out = {**rec, "llm_label": label}
        labeled.append(rec_out)
        print(f"[{i+1}/{len(records)}] L{rec['layer']} E{rec['expert_id']} F{rec['feature_id']}: {label}")
        time.sleep(sleep_s)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in labeled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(labeled)} features labeled -> {out_path}")
    return labeled


def load_top_feature_records(available, layer_key, top_k=6, metric="mean_activation"):
    import numpy as np

    out = {}
    present = available.get(layer_key, {})
    for expert_id, path in present.items():
        with open(path) as f:
            d = json.load(f)
        if not d.get("features"):
            continue
        scored = []
        for feat in d["features"]:
            acts = [ex["activation"] for ex in feat["examples"]]
            feat = dict(feat) 
            feat["mean_activation"] = float(np.mean(acts)) if acts else 0.0
            scored.append(feat)
        scored.sort(key=lambda r: r[metric], reverse=True)
        top = scored[:top_k]
        if top:
            out[expert_id] = top
    return out
