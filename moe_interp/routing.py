import torch
import numpy as np
import pickle
import os
from collections import defaultdict

from .data import chunk_tokenize


@torch.no_grad()
def run_encoder_and_record(model, input_ids, recorder):
    recorder.clear()
    attn = torch.ones_like(input_ids)
    model.encoder(
        input_ids=input_ids.to(model.device),
        attention_mask=attn.to(model.device),
    )
    return {k: v["topk_idx"].squeeze(-1).squeeze(0) for k, v in recorder.records.items()}


@torch.no_grad()
def run_encoder_and_record_all_layers(model, input_ids, recorder):
    recorder.clear()
    attn = torch.ones_like(input_ids)
    model.encoder(
        input_ids=input_ids.to(model.device),
        attention_mask=attn.to(model.device),
    )
    return {k: v["topk_idx"].reshape(-1) for k, v in recorder.records.items()}


def collect_domain_routing_stats(model, tokenizer, cfg, domain_data, enc_layer_keys, recorder):
    n_experts = cfg["n_routed_experts"]
    expert_counts = {d: {lk: np.zeros(n_experts) for lk in enc_layer_keys} for d in domain_data}
    repeat_counts = {d: {lk: [0, 0] for lk in enc_layer_keys} for d in domain_data}

    for domain, texts in domain_data.items():
        chunks = chunk_tokenize(texts, tokenizer)
        for ids in chunks:
            layer_idx = run_encoder_and_record(model, ids, recorder)
            for lk in enc_layer_keys:
                idx = layer_idx[lk]
                valid = idx[idx >= 0]
                for e in valid.tolist():
                    expert_counts[domain][lk][int(e)] += 1
                if idx.shape[0] > 1:
                    same = (idx[1:] == idx[:-1]) & (idx[1:] >= 0) & (idx[:-1] >= 0)
                    repeat_counts[domain][lk][0] += same.sum().item()
                    repeat_counts[domain][lk][1] += (idx.shape[0] - 1)

    return expert_counts, repeat_counts


def collect_tagged_activations(model, tokenizer, cfg, domain, texts, layer_key,
                                routing_recorder, act_recorder, seq_len=None, max_chunks=None):
    from .data import SEQ_LEN, MAX_CHUNKS_PER_DOMAIN
    seq_len = seq_len or SEQ_LEN
    max_chunks = max_chunks or MAX_CHUNKS_PER_DOMAIN

    chunks = chunk_tokenize(texts, tokenizer, seq_len=seq_len, max_chunks=max_chunks)
    meta = {e: [] for e in range(cfg["n_routed_experts"])}
    buf_ptr = {e: 0 for e in range(cfg["n_routed_experts"])}

    for doc_id, ids in enumerate(chunks):
        routing_recorder.clear()
        idx_by_layer = run_encoder_and_record_all_layers(model, ids, routing_recorder)
        idx = idx_by_layer[layer_key].tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids[0])

        per_expert_tokens = {e: [] for e in range(cfg["n_routed_experts"])}
        for pos, (tok, e) in enumerate(zip(tokens, idx)):
            if e >= 0:
                per_expert_tokens[e].append((tok, pos))

        for e, tok_positions in per_expert_tokens.items():
            buf = act_recorder.buffers[e]
            if len(buf) > buf_ptr[e]:
                n_new = buf[buf_ptr[e]].shape[0]
                buf_ptr[e] += 1
            else:
                n_new = 0
            if n_new != len(tok_positions):
                continue 
            for tok, pos in tok_positions:
                meta[e].append({"domain": domain, "doc_id": doc_id, "pos": pos, "token": tok})
    return meta


def collect_and_cache_expert_activations(model, tokenizer, cfg, domain_data, layer_key,
                                          routing_recorder, act_recorder, cache_dir,
                                          seq_len=None, max_chunks=None):
    meta_dir = os.path.join(cache_dir, f"layer_{layer_key}_meta")
    os.makedirs(meta_dir, exist_ok=True)

    for domain, texts in domain_data.items():
        meta = collect_tagged_activations(
            model, tokenizer, cfg, domain, texts, layer_key,
            routing_recorder, act_recorder, seq_len=seq_len, max_chunks=max_chunks,
        )
        act_recorder.flush_to_disk(cache_dir, tag=f"_{domain.replace(' ', '_')}")
        for e, rows in meta.items():
            if not rows:
                continue
            with open(os.path.join(meta_dir, f"expert_{e}_{domain.replace(' ', '_')}.pkl"), "wb") as f:
                pickle.dump(rows, f)
        print(f"{domain}: done")


def load_expert_data(layer_key, expert_idx, domains, cache_dir, meta_dir):
    layer_dir = os.path.join(cache_dir, f"layer_{layer_key}")
    acts, meta = [], []
    missing, mismatched = [], []
    for domain in domains:
        tag = domain.replace(" ", "_")
        act_path = os.path.join(layer_dir, f"expert_{expert_idx}_{tag}.npy")
        meta_path = os.path.join(meta_dir, f"expert_{expert_idx}_{tag}.pkl")
        if not (os.path.exists(act_path) and os.path.exists(meta_path)):
            missing.append(domain)
            continue
        arr = np.load(act_path)
        with open(meta_path, "rb") as f:
            rows = pickle.load(f)
        if arr.shape[0] != len(rows):
            mismatched.append(domain)
            continue
        acts.append(arr)
        meta.extend(rows)

    if not acts:
        raise ValueError(
            f"No usable activation data for expert {expert_idx} @ {layer_key}. "
            f"Missing files for domains: {missing}. Shape-mismatched: {mismatched}. "
            f"This expert may have received ~0 routed tokens, or collection needs re-running."
        )
    if missing or mismatched:
        print(f"Note: expert {expert_idx} — skipped missing={missing}, mismatched={mismatched}")
    return np.concatenate(acts, axis=0), meta


def collect_language_routing(model, tokenizer, lang_data, layer_key, routing_recorder,
                              seq_len=128, max_chunks=100):
    lang_expert_counts = {}
    for lang_name, texts in lang_data.items():
        counts = defaultdict(int)
        chunks = chunk_tokenize(texts, tokenizer, seq_len=seq_len, max_chunks=max_chunks)
        for ids in chunks:
            idx_by_layer = run_encoder_and_record_all_layers(model, ids, routing_recorder)
            idx = idx_by_layer[layer_key].tolist()
            for e in idx:
                if e >= 0:
                    counts[e] += 1
        lang_expert_counts[lang_name] = dict(counts)
        total = sum(counts.values())
        print(f"{lang_name}: {total} routed tokens")
    return lang_expert_counts
