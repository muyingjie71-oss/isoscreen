"""Split form of the two production models: molecule encoder (run once per molecule) + pair head (run per solute-solvent pair).

Written with plain tensor ops (gather, matmul, elementwise) so the same graphs can be exported to ONNX and run in the browser.
Every function is checked against the production predictors (predict_dmpnn.py, predict_ghgnn.py) in verify_split.py.

D-MPNN (Layer A, A_JOINT):   ln g = MLP([hi, hs, hi-hs, hi*hs, silu(W_T t + b), 0]) * y_std + y_mean,  t = (1000/T - t_mean)/t_std
GH-GNN (IL-extended):        per molecule: graphnet1 -> GraphNorm -> graphnet2 -> GraphNorm -> mean pool -> [xg, u] (226)
                             per pair: 2-node system graph (NNConv + GRU, every edge carries inter_hb), two MLP heads -> A, B;
                             ln g = A + B / T  (T in K)
"""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False)
F = torch.nn.functional

# ----------------------------------------------------------------------------------------------- D-MPNN
class DMPNNEncoder(torch.nn.Module):
    """Inputs for a batch of G molecules: atom (N,50), bond (E,14), src (E), rev (E), dst_inc (N,E) one-hot incidence,
    graph_inc (G,N) one-hot membership. Output (G,256)."""
    def __init__(self, p, depth=4):
        super().__init__()
        t = lambda k: torch.tensor(p[k])
        self.Wb, self.bb = t('shared_encoder/bond_input/kernel'), t('shared_encoder/bond_input/bias')
        self.Wm = t('shared_encoder/bond_message/kernel')
        self.Wa, self.ba = t('shared_encoder/atom_output/kernel'), t('shared_encoder/atom_output/bias')
        self.Wr, self.br = t('shared_encoder/size_aware_readout/kernel'), t('shared_encoder/size_aware_readout/bias')
        self.depth = depth
    def forward(self, atom, bond, src, rev, dst_inc, graph_inc):
        h0 = F.silu(torch.cat([atom[src], bond], -1) @ self.Wb + self.bb)
        h = h0
        for _ in range(self.depth - 1):
            incoming = dst_inc @ h
            h = F.silu(h0 + (incoming[src] - h[rev]) @ self.Wm)
        incoming = dst_inc @ h
        nodes = F.silu(torch.cat([atom, incoming], -1) @ self.Wa + self.ba)
        pooled = graph_inc @ nodes
        count = graph_inc.sum(1, keepdim=True)
        feats = torch.cat([pooled / torch.clamp(count, min=1), pooled, torch.log1p(count)], -1)
        return F.silu(feats @ self.Wr + self.br)

class DMPNNHead(torch.nn.Module):
    """hi (1,256) solute, hs (S,256) solvents, t (K,) standardised temperatures -> raw output (S,K) before y-scaling.
    The first fusion layer is split by input block; the zero block contributes nothing."""
    def __init__(self, p):
        super().__init__()
        t = lambda k: torch.tensor(p[k])
        W = t('fusion_512/kernel'); H = 256
        self.W_i, self.W_s, self.W_d, self.W_p, self.W_t = W[0:H], W[H:2*H], W[2*H:3*H], W[3*H:4*H], W[4*H:4*H+32]
        self.b1 = t('fusion_512/bias'); self.W2, self.b2 = t('fusion_256/kernel'), t('fusion_256/bias')
        self.W3, self.b3 = t('ln_gamma/kernel'), t('ln_gamma/bias')
        self.Wt, self.bt = t('temperature/kernel'), t('temperature/bias')
    def forward(self, hi, hs, tz):
        ht = F.silu(tz[:, None] @ self.Wt + self.bt)                          # (K,32)
        base = (hi @ self.W_i) + (hs @ self.W_s) + ((hi - hs) @ self.W_d) + ((hi * hs) @ self.W_p) + self.b1   # (S,512)
        x = F.silu(base[:, None, :] + (ht @ self.W_t)[None, :, :])             # (S,K,512)
        x = F.silu(x @ self.W2 + self.b2)
        return (x @ self.W3 + self.b3)[..., 0]                                 # (S,K)

def dmpnn_graph_batch(graphs):
    """graphs: list of dicts from graphs.graph(smiles). Returns tensors for DMPNNEncoder."""
    atoms, bonds, src, dst, rev, gidx = [], [], [], [], [], []; na = ne = 0
    for gi, g in enumerate(graphs):
        atoms.append(g['atom']); bonds.append(g['bond']); src.append(g['src'] + na); dst.append(g['dst'] + na); rev.append(g['rev'] + ne)
        gidx.append(np.full(len(g['atom']), gi)); na += len(g['atom']); ne += len(g['src'])
    atom = torch.tensor(np.concatenate(atoms)); bond = torch.tensor(np.concatenate(bonds).reshape(-1, 14))
    src = torch.tensor(np.concatenate(src), dtype=torch.long); dst = torch.tensor(np.concatenate(dst), dtype=torch.long)
    rev = torch.tensor(np.concatenate(rev), dtype=torch.long); gidx = torch.tensor(np.concatenate(gidx), dtype=torch.long)
    dst_inc = torch.zeros(na, ne); dst_inc[dst, torch.arange(ne)] = 1
    graph_inc = torch.zeros(len(graphs), na); graph_inc[gidx, torch.arange(na)] = 1
    return atom, bond, src, rev, dst_inc, graph_inc

# ----------------------------------------------------------------------------------------------- GH-GNN
def _lin(m): return m.weight.T.contiguous(), m.bias

class GHGNNEncoder(torch.nn.Module):
    """Inputs for a batch of G molecules: x (N,37), edge_attr (E,9), row/col (E), u (G,3) = [ap, bp, topopsa],
    node_inc (G,N) one-hot membership, col_inc (N,E) one-hot (edge -> target node), edge_graph_inc (G,E) one-hot.
    Output (G,226) = [mean-pooled nodes (113), global features u (113)]."""
    def __init__(self, model):
        super().__init__()
        self.L = {}
        for name in ('graphnet1', 'graphnet2'):
            gn = getattr(model, name)
            self.L[name] = {'e': [_lin(gn.edge_model.edge_mlp[0]), _lin(gn.edge_model.edge_mlp[2])],
                            'n': [_lin(gn.node_model.node_mlp[0]), _lin(gn.node_model.node_mlp[2])],
                            'g': [_lin(gn.global_model.global_mlp[0]), _lin(gn.global_model.global_mlp[2])]}
        self.N = {k: (getattr(model, k).weight, getattr(model, k).bias, getattr(model, k).mean_scale, getattr(model, k).eps) for k in ('gnorm1', 'gnorm2')}
    @staticmethod
    def mlp(x, ws): (W1, b1), (W2, b2) = ws; return F.relu(x @ W1 + b1) @ W2 + b2
    def meta(self, L, x, ea, u, row, col, node_inc, col_inc, edge_graph_inc):
        node_graph = node_inc.argmax(0)
        ea = self.mlp(torch.cat([x[row], x[col], ea, u[node_graph[row]]], 1), L['e'])
        x = self.mlp(torch.cat([x, col_inc @ ea, u[node_graph]], 1), L['n'])
        nmean = (node_inc @ x) / torch.clamp(node_inc.sum(1, keepdim=True), min=1)
        emean = (edge_graph_inc @ ea) / torch.clamp(edge_graph_inc.sum(1, keepdim=True), min=1)
        u = self.mlp(torch.cat([u, nmean, emean], 1), L['g'])
        return x, ea, u
    def norm(self, k, x, node_inc):
        w, b, ms, eps = self.N[k]; node_graph = node_inc.argmax(0); cnt = torch.clamp(node_inc.sum(1, keepdim=True), min=1)
        mean = (node_inc @ x) / cnt
        out = x - mean[node_graph] * ms
        var = (node_inc @ (out * out)) / cnt
        return w * out / torch.sqrt(var + eps)[node_graph] + b
    def forward(self, x, ea, row, col, u, node_inc, col_inc, edge_graph_inc):
        x, ea, u = self.meta(self.L['graphnet1'], x, ea, u, row, col, node_inc, col_inc, edge_graph_inc)
        x = self.norm('gnorm1', x, node_inc)
        x, ea, u = self.meta(self.L['graphnet2'], x, ea, u, row, col, node_inc, col_inc, edge_graph_inc)
        x = self.norm('gnorm2', x, node_inc)
        xg = (node_inc @ x) / torch.clamp(node_inc.sum(1, keepdim=True), min=1)
        return torch.cat([xg, u], 1)

class GHGNNHead(torch.nn.Module):
    """Pair head. z1 (S,226) solvents, z2 (S,226) solutes, hb (S,) inter_hb. Returns A, B (S,) with ln g = A + B/T[K].
    System graph: nodes [solvent, solute]; edges solvent->solute, solute->solvent and two self loops, all with edge feature inter_hb.
    NNConv (aggr add) + root weight + bias, ReLU, one GRU step with the projected features as hidden state."""
    def __init__(self, model):
        super().__init__()
        gc = model.global_conv1
        self.Wp, self.bp = _lin(gc.project_node_feats[0])
        (self.We1, self.be1), (self.We2, self.be2) = _lin(gc.gnn_layer.nn[0]), _lin(gc.gnn_layer.nn[2])
        self.Wroot = gc.gnn_layer.lin.weight.T.contiguous(); self.bconv = gc.gnn_layer.bias
        g = gc.gru; self.Wih, self.Whh, self.bih, self.bhh = g.weight_ih_l0.T.contiguous(), g.weight_hh_l0.T.contiguous(), g.bias_ih_l0, g.bias_hh_l0
        self.heads = [[_lin(getattr(model, f'mlp{i}{h}')) for i in (1, 2, 3)] for h in ('a', 'b')]
        self.D = 226
    def edge_weight(self, hb):                                    # (S,) -> (S,226,226)
        h = F.relu(hb[:, None] @ self.We1 + self.be1)
        return (h @ self.We2 + self.be2).view(-1, self.D, self.D)
    def gru(self, x, h):
        gi = x @ self.Wih + self.bih; gh = h @ self.Whh + self.bhh
        i_r, i_z, i_n = gi.chunk(3, 1); h_r, h_z, h_n = gh.chunk(3, 1)
        r = torch.sigmoid(i_r + h_r); z = torch.sigmoid(i_z + h_z); n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h
    def forward(self, z1, z2, hb):
        p1 = F.relu(z1 @ self.Wp + self.bp); p2 = F.relu(z2 @ self.Wp + self.bp)
        W = self.edge_weight(hb)
        msg1 = torch.bmm((p2 + p1)[:, None, :], W)[:, 0]          # solvent node receives from solute and itself
        msg2 = torch.bmm((p1 + p2)[:, None, :], W)[:, 0]          # solute node receives from solvent and itself
        c1 = F.relu(msg1 + p1 @ self.Wroot + self.bconv); c2 = F.relu(msg2 + p2 @ self.Wroot + self.bconv)
        n1 = self.gru(c1, p1); n2 = self.gru(c2, p2)
        xg = torch.cat([n1, n2], 1)
        out = []
        for (l1, l2, l3) in self.heads:
            a = F.relu(xg @ l1[0] + l1[1]); a = F.relu(a @ l2[0] + l2[1]); out.append((a @ l3[0] + l3[1])[:, 0])
        Cn, Dn = out
        C = Cn * (29.1066045841922 + 2.91058307855874) - 2.91058307855874
        D = Dn * (4.8134735224616 + 3.97198033957307) - 3.97198033957307
        T1, T2 = 303.15, 363.15
        B = D * 2 * T1 * T2 / (T2 - T1)
        A = C - B * (T1 + T2) / (2 * T1 * T2)
        return A, B
