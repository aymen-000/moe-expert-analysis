import matplotlib.pyplot as plt

STOPWORDS = set("""
a an the is are was were be been being of to in on at for with by from as it
its this that these those and or but if then else not no yes do does did
have has had will would can could should shall may might i you he she we
they them his her their our your my me him us
""".split())

SPECIAL_TOKENS = {"<unk>", "<pad>", "<s>", "</s>", "<extra_id_0>", "<extra_id_1>"}


def is_meaningful_token(tok, min_alpha_chars=2, min_alpha_ratio=0.5):
    raw = tok.replace("▁", "").strip()
    if not raw or raw in SPECIAL_TOKENS:
        return False
    alpha_chars = sum(ch.isalnum() for ch in raw)
    if alpha_chars < min_alpha_chars:
        return False
    if alpha_chars / len(raw) < min_alpha_ratio:
        return False
    if raw.lower() in STOPWORDS:
        return False
    return True


def expert_colors(n):
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [tuple(int(255 * c) for c in cmap(i % cmap.N)[:3]) for i in range(n)]
