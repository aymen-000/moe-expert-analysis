import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.d_in, self.d_hidden = d_in, d_hidden
        self.W_enc = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)  
        self.normalize_decoder()

    def normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True) + 1e-8)

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


def train_sae(X_np, d_hidden_mult=2, l1_coef=3e-3, lr=1e-3, epochs=30, batch_size=512, device="cuda"):
    X = torch.tensor(X_np, dtype=torch.float32)
    mean, std = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    X = (X - mean) / std  # normalize; store mean/std to invert later if needed

    d_in = X.shape[1]
    sae = SparseAutoencoder(d_in, d_in * d_hidden_mult).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    n = X.shape[0]
    fired_ever = torch.zeros(sae.d_hidden, dtype=torch.bool)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_mse, total_l1 = 0.0, 0.0
        for i in range(0, n, batch_size):
            batch = X[perm[i:i + batch_size]].to(device)
            x_hat, f = sae(batch)
            mse = F.mse_loss(x_hat, batch)
            l1 = f.abs().mean()
            loss = mse + l1_coef * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()  
            fired_ever |= (f.detach().cpu() > 0).any(dim=0)
            total_mse += mse.item() * batch.shape[0]
            total_l1 += l1.item() * batch.shape[0]
        if epoch % 5 == 0 or epoch == epochs - 1:
            dead = (~fired_ever).sum().item()
            print(f"epoch {epoch:3d}  mse={total_mse/n:.4f}  l1={total_l1/n:.4f}  "
                  f"dead_features={dead}/{sae.d_hidden}")

    return sae, {"mean": mean, "std": std}


@torch.no_grad()
def get_feature_activations(sae, X_np, norm_stats, device="cuda", batch_size=1024):
    X = torch.tensor(X_np, dtype=torch.float32)
    X = (X - norm_stats["mean"]) / norm_stats["std"]
    feats = []
    for i in range(0, X.shape[0], batch_size):
        f = sae.encode(X[i:i + batch_size].to(device)).cpu()
        feats.append(f)
    return torch.cat(feats, dim=0)


def rank_features_by_importance(F_acts, top_n=40, min_density=0.00001, max_density=0.8):
    density = (F_acts > 0).float().mean(dim=0)
    mass = F_acts.sum(dim=0)
    eligible = (density >= min_density) & (density <= max_density)
    eligible_idx = eligible.nonzero().squeeze(-1)
    if eligible_idx.numel() == 0:
        return []
    ranked = eligible_idx[torch.argsort(mass[eligible_idx], descending=True)]
    return ranked[:top_n].tolist()
