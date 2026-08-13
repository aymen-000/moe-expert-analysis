from datasets import load_dataset

PILE_DOMAINS = [
    "ArXiv", "Github", "PubMed Abstracts", "PhilPapers",
    "Wikipedia (en)", "DM Mathematics", "Gutenberg (PG-19)", "StackExchange",
]

LANGUAGES = {
    "English": "en", "Arabic": "ar",
    "French": "fr", "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Simplified)": "zh",
}

SEQ_LEN = 128
MAX_CHUNKS_PER_DOMAIN = 60


def load_pile_domains(domains=PILE_DOMAINS, n_docs=60, min_chars=200):
    pile = load_dataset("NeelNanda/pile-10k", split="train")

    def domain_texts(domain):
        texts = []
        for ex in pile:
            if ex["meta"]["pile_set_name"] == domain and len(ex["text"]) >= min_chars:
                texts.append(ex["text"])
                if len(texts) >= n_docs:
                    break
        return texts

    domain_data = {d: domain_texts(d) for d in domains}
    for d, texts in domain_data.items():
        print(f"{d}: {len(texts)} docs")
    return domain_data


def load_language_samples(n_docs=100, min_chars=100, languages=LANGUAGES):
    lang_data = {}
    for name, code in languages.items():
        try:
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{code}", split="train", streaming=True)
            texts = []
            for ex in ds:
                if len(ex["text"]) >= min_chars:
                    texts.append(ex["text"][:800])
                if len(texts) >= n_docs:
                    break
            lang_data[name] = texts
            print(f"{name} ({code}): {len(texts)} docs")
        except Exception as e:
            print(f"{name} ({code}): FAILED — {e}")
    return lang_data


def chunk_tokenize(texts, tokenizer, seq_len=SEQ_LEN, max_chunks=MAX_CHUNKS_PER_DOMAIN):
    chunks = []
    for t in texts:
        ids = tokenizer(t, truncation=True, max_length=seq_len, return_tensors="pt")["input_ids"]
        if ids.shape[1] < 8:
            continue
        chunks.append(ids)
        if len(chunks) >= max_chunks:
            break
    return chunks
