"""GNN encoders and self-supervised objectives (paper Sections 4.4, 7.2, 7.3).

GCN / GraphSAGE two-layer encoders trained with VGAE (primary) or DGI.
`ablate=True` removes the graph from the encoder (A = I, baseline B4) while
keeping the same objective, which isolates the effect of message passing.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv, SAGEConv, VGAE, DeepGraphInfomax
from torch_geometric.utils import negative_sampling, to_undirected, subgraph

torch.set_num_threads(max(1, torch.get_num_threads()))


def make_conv(kind, i, o, normalize=False):
    if kind == "gcn":
        return GCNConv(i, o)
    return SAGEConv(i, o, aggr="mean", normalize=normalize)


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out=32, kind="gcn", variational=False, dropout=0.0):
        super().__init__()
        self.variational, self.dropout = variational, dropout
        # Eq. (5): GraphSAGE hidden layer is followed by L2 normalization.
        self.conv1 = make_conv(kind, in_dim, hidden, normalize=(kind == "sage"))
        self.conv_mu = make_conv(kind, hidden, out)
        if variational:
            self.conv_logstd = make_conv(kind, hidden, out)
        self.act = nn.ReLU() if variational else nn.PReLU(out)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = nn.functional.dropout(h, self.dropout, self.training)
        if self.variational:
            return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)
        return self.act(self.conv_mu(h, edge_index))


def split_edges(edge_index, n, val_frac=0.05, seed=0):
    """Hold out a fraction of undirected edges for link-prediction validation."""
    g = torch.Generator().manual_seed(seed)
    und = edge_index[:, edge_index[0] < edge_index[1]]
    perm = torch.randperm(und.size(1), generator=g)
    n_val = int(val_frac * und.size(1))
    val_pos, tr = und[:, perm[:n_val]], und[:, perm[n_val:]]
    val_neg = negative_sampling(edge_index, num_nodes=n, num_neg_samples=n_val)
    return to_undirected(tr, num_nodes=n), val_pos, val_neg


def link_auc(z, pos, neg):
    s = torch.cat([(z[pos[0]] * z[pos[1]]).sum(1), (z[neg[0]] * z[neg[1]]).sum(1)])
    y = np.r_[np.ones(pos.size(1)), np.zeros(neg.size(1))]
    return roc_auc_score(y, s.detach().numpy())


def train_gnn(X, edge_index, train_idx, kind="gcn", objective="vgae", dim=32,
              hidden=64, lr=0.01, weight_decay=5e-4, dropout=0.0, epochs=200,
              patience=20, seed=0, ablate=False, verbose=False, eval_every=5):
    """Train an encoder without labels and return embeddings for all nodes.

    GCN is trained transductively on the full similarity graph; GraphSAGE is
    trained inductively on the training-node subgraph and then applied to the
    full graph so that validation / test customers attach to training ones.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = X.shape[0]
    x = torch.tensor(X, dtype=torch.float32)
    tr_ei, val_pos, val_neg = split_edges(edge_index, n, seed=seed)

    if kind == "sage":  # inductive: train only on edges among training customers
        mask = torch.zeros(n, dtype=torch.bool)
        mask[torch.as_tensor(train_idx)] = True
        tr_ei, _ = subgraph(mask, tr_ei, num_nodes=n)

    empty = torch.empty(2, 0, dtype=torch.long)
    enc_ei_train = empty if ablate else tr_ei          # B4: A = I
    enc_ei_full = empty if ablate else edge_index

    enc = Encoder(X.shape[1], hidden, dim, kind, variational=(objective == "vgae"),
                  dropout=dropout)
    if objective == "vgae":
        model = VGAE(enc)
    else:
        model = DeepGraphInfomax(
            dim, enc,
            summary=lambda z, *a, **k: torch.sigmoid(z.mean(0)),
            corruption=lambda x, ei: (x[torch.randperm(x.size(0))], ei))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        if objective == "vgae":
            z = model.encode(x, enc_ei_train)
            loss = model.recon_loss(z, tr_ei) + model.kl_loss() / n
        else:
            pos, neg, s = model(x, enc_ei_train)
            loss = model.loss(pos, neg, s)
        loss.backward()
        opt.step()
        if ep % eval_every and ep != epochs - 1:
            continue

        model.eval()
        with torch.no_grad():
            z = embed(model, objective, x, enc_ei_full)
        auc = link_auc(z, val_pos, val_neg)
        hist.append((ep, loss.item(), auc))
        if auc > best + 1e-4:
            best, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += eval_every
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Z = embed(model, objective, x, enc_ei_full).numpy()
    if verbose:
        print(f"  {kind}/{objective} ablate={ablate} epochs={ep + 1} val_auc={best:.3f}")
    return Z, {"val_auc": best, "epochs": ep + 1, "history": hist}, model


def embed(model, objective, x, edge_index):
    if objective == "vgae":
        return model.encoder(x, edge_index)[0]  # posterior mean
    return model.encoder(x, edge_index)


class AutoEncoder(nn.Module):
    """B3: MLP autoencoder with the same widths as the GNN encoder (no graph)."""

    def __init__(self, d, hidden=64, dim=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.dec = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_autoencoder(X, train_idx, val_idx, dim=32, hidden=64, lr=0.005,
                      epochs=200, patience=20, batch=512, seed=0):
    torch.manual_seed(seed)
    x = torch.tensor(X, dtype=torch.float32)
    ae = AutoEncoder(X.shape[1], hidden, dim)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=5e-4)
    xtr, xva = x[train_idx], x[val_idx]
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        ae.train()
        for b in torch.randperm(len(xtr)).split(batch):
            opt.zero_grad()
            loss = ((ae(xtr[b]) - xtr[b]) ** 2).mean()
            loss.backward()
            opt.step()
        ae.eval()
        with torch.no_grad():
            v = float(((ae(xva) - xva) ** 2).mean())
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(ae.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    ae.load_state_dict(best_state)
    with torch.no_grad():
        return ae.enc(x).numpy()
