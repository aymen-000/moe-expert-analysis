import pandas as pd

from .utils import is_meaningful_token


def collect_feature_examples(feat_idx, F_acts, X_meta, get_doc_tokens_fn, k=20, window=8, search_pool=400):
    col = F_acts[:, feat_idx]
    n_firing = (col > 0).sum().item()
    if n_firing == 0:
        return []
    pool_size = min(search_pool, n_firing)
    top_vals, top_idx = col.topk(pool_size)

    examples = []
    for val, i in zip(top_vals.tolist(), top_idx.tolist()):
        m = X_meta[i]
        if not is_meaningful_token(m["token"]):
            continue
        tokens = get_doc_tokens_fn(m["domain"], m["doc_id"])
        pos = m["pos"]
        lo, hi = max(0, pos - window), min(len(tokens), pos + window + 1)
        ctx_tokens = [t.replace("▁", " ") for t in tokens[lo:hi]]
        ctx_acts = [0.0] * len(ctx_tokens)
        ctx_acts[pos - lo] = val
        examples.append({
            "activation": round(val, 4),
            "domain": m["domain"],
            "token": m["token"].replace("▁", " ").strip(),
            "context_tokens": ctx_tokens,
            "context_acts": ctx_acts,
            "center_idx": pos - lo,
        })
        if len(examples) >= k:
            break
    return examples


def build_dashboard_data(candidate_feats, F_acts, X_meta, density, max_act,
                          get_doc_tokens_fn, k=20, window=8, min_examples=4):
    data = []
    for feat_idx in candidate_feats:
        examples = collect_feature_examples(feat_idx, F_acts, X_meta, get_doc_tokens_fn, k=k, window=window)
        if len(examples) < min_examples:
            continue
        dom_counts = pd.Series([e["domain"] for e in examples]).value_counts(normalize=True)
        data.append({
            "feature": int(feat_idx),
            "density": float(density[feat_idx]),
            "max_act": float(max_act[feat_idx]),
            "top_domain": dom_counts.index[0],
            "top_domain_share": round(float(dom_counts.iloc[0]), 3),
            "n_examples": len(examples),
            "examples": examples,
        })
    data.sort(key=lambda d: d["top_domain_share"], reverse=True)
    return data


def collect_examples_for_llm(feat_idx, F_acts, X_meta, get_doc_tokens_fn,
                              n_examples=150, window=10, search_pool=1000):
    col = F_acts[:, feat_idx]
    n_firing = (col > 0).sum().item()
    if n_firing == 0:
        return []
    pool_size = min(search_pool, n_firing)
    top_vals, top_idx = col.topk(pool_size)

    examples = []
    for val, i in zip(top_vals.tolist(), top_idx.tolist()):
        m = X_meta[i]
        if not is_meaningful_token(m["token"]):
            continue
        tokens = get_doc_tokens_fn(m["domain"], m["doc_id"])
        pos = m["pos"]
        lo, hi = max(0, pos - window), min(len(tokens), pos + window + 1)
        ctx = "".join(t.replace("▁", " ") for t in tokens[lo:hi])
        target_tok = tokens[pos].replace("▁", " ").strip()
        examples.append({
            "activation": round(val, 4),
            "domain": m["domain"],
            "target_token": target_tok,
            "context": ctx.strip(),
        })
        if len(examples) >= n_examples:
            break
    return examples


def token_routing_profile(expert_idx, meta_dir, domains, top_n=25):
    import os
    import pickle

    all_rows = []
    for domain in domains:
        p = os.path.join(meta_dir, f"expert_{expert_idx}_{domain.replace(' ', '_')}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                all_rows.extend(pickle.load(f))
    if not all_rows:
        return None, None
    domain_counts = pd.Series([r["domain"] for r in all_rows]).value_counts()
    token_counts = pd.Series([r["token"] for r in all_rows]).value_counts().head(top_n)
    return domain_counts, token_counts


def build_llm_prompt(feature_record):
    header = (
        f"Layer: {feature_record['layer']} | Expert: {feature_record['expert_id']} | "
        f"Feature: {feature_record['feature_id']} | Density: {feature_record['density']}\n"
        f"Below are text snippets. In each, the TARGET token is the one that "
        f"strongly activated this feature (shown in [brackets] within its context). "
        f"Based on these examples, describe in one sentence what concept, pattern, "
        f"or property this feature appears to detect.\n\n"
    )
    lines = []
    for ex in feature_record["examples"]:
        marked = ex["context"].replace(ex["target_token"], f"[{ex['target_token']}]", 1)
        lines.append(f"(act={ex['activation']:.2f}, domain={ex['domain']}) {marked}")
    return header + "\n".join(lines)
