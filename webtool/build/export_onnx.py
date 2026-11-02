"""Export the pair heads (and solvent-side preparation) of both models to ONNX for onnxruntime-web, one file per seed and part.
Weights are stored as float16 initialisers and cast to float32 inside the graph (halves the download; effect measured in tests).
Also writes weights_js.bin (float16) with what the JavaScript side needs: both solute encoders and the GH-GNN edge network.
Usage: python export_onnx.py <out dir>"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import MODELS, PARAMS, ASSETS
import mol2graph as M
from GHGNN_architecture import GHGNN_model
from run_ghgnn_job import HP
F = torch.nn.functional
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ASSETS; WDT = sys.argv[2] if len(sys.argv) > 2 else 'f32'; ONX = OUT / ('onnx' if WDT == 'f16' else 'onnx32'); ONX.mkdir(parents=True, exist_ok=True)
SEEDS = ['seed_20261007', 'seed_20261008', 'seed_20261009']
h16 = lambda a: torch.tensor(np.asarray(a, np.float32)).half() if WDT == 'f16' else torch.tensor(np.asarray(a, np.float32))

class Base(torch.nn.Module):
    def w(self, name, a): self.register_buffer(name, h16(a))
    def g(self, name): return getattr(self, name).float()

class DPrep(Base):
    def __init__(s, p): super().__init__(); s.w('Wsd', p['fusion_512/kernel'][256:512] - p['fusion_512/kernel'][512:768])
    def forward(s, hs): return hs @ s.g('Wsd')

class DHead(Base):
    def __init__(s, p, sc):
        super().__init__(); K = p['fusion_512/kernel']; H = 256
        s.w('Wid', K[0:H] + K[2*H:3*H]); s.w('Wp', K[3*H:4*H]); s.w('Wt', K[4*H:4*H+32]); s.w('b1', p['fusion_512/bias'])
        s.w('W2', p['fusion_256/kernel']); s.w('b2', p['fusion_256/bias']); s.w('W3', p['ln_gamma/kernel']); s.w('b3', p['ln_gamma/bias'])
        s.w('tW', p['temperature/kernel']); s.w('tb', p['temperature/bias']); s.ys, s.ym = float(sc['y_std']), float(sc['y_mean'])
    def forward(s, hi, hs, sp, tz):                       # hi (1,256), hs (n,256), sp (n,512), tz (K,)
        base = hi @ s.g('Wid') + sp + (hi * hs) @ s.g('Wp') + s.g('b1')                     # (n,512)
        ht = F.silu(tz[:, None] @ s.g('tW') + s.g('tb')) @ s.g('Wt')                        # (K,512)
        x = F.silu(base[:, None, :] + ht[None, :, :])                                        # (n,K,512)
        x = F.silu(x @ s.g('W2') + s.g('b2'))
        return (x @ s.g('W3') + s.g('b3'))[..., 0] * s.ys + s.ym                             # (n,K)

class GPrep(Base):
    def __init__(s, m): super().__init__(); s.w('Wp', m.global_conv1.project_node_feats[0].weight.detach().T); s.w('bp', m.global_conv1.project_node_feats[0].bias.detach())
    def forward(s, z): return F.relu(z @ s.g('Wp') + s.g('bp'))

class GHead(Base):
    def __init__(s, m):
        super().__init__(); gc = m.global_conv1
        s.w('Wroot', gc.gnn_layer.lin.weight.detach().T); s.w('bconv', gc.gnn_layer.bias.detach())
        s.w('Wih', gc.gru.weight_ih_l0.detach().T); s.w('Whh', gc.gru.weight_hh_l0.detach().T); s.w('bih', gc.gru.bias_ih_l0.detach()); s.w('bhh', gc.gru.bias_hh_l0.detach())
        for h in ('a', 'b'):
            for i in (1, 2, 3):
                l = getattr(m, f'mlp{i}{h}'); s.w(f'W{i}{h}', l.weight.detach().T); s.w(f'b{i}{h}', l.bias.detach())
    def gru(s, x, h, gh):
        gi = x @ s.g('Wih') + s.g('bih'); D = 226
        r = torch.sigmoid(gi[:, :D] + gh[:, :D]); z = torch.sigmoid(gi[:, D:2*D] + gh[:, D:2*D]); n = torch.tanh(gi[:, 2*D:] + r * gh[:, 2*D:])
        return (1 - z) * n + z * h
    def forward(s, p1, p2, Wv):                          # p1 (n,226) solvents, p2 (1,226) solute, Wv (226,226) edge weight for this inter_hb
        m = (p1 + p2) @ Wv
        c1 = F.relu(m + p1 @ s.g('Wroot') + s.g('bconv')); c2 = F.relu(m + p2 @ s.g('Wroot') + s.g('bconv'))
        n1 = s.gru(c1, p1, p1 @ s.g('Whh') + s.g('bhh')); n2 = s.gru(c2, p2.expand_as(p1), p2 @ s.g('Whh') + s.g('bhh'))
        xg = torch.cat([n1, n2], 1); out = []
        for h in ('a', 'b'):
            a = F.relu(xg @ s.g(f'W1{h}') + s.g(f'b1{h}')); a = F.relu(a @ s.g(f'W2{h}') + s.g(f'b2{h}')); out.append((a @ s.g(f'W3{h}') + s.g(f'b3{h}'))[:, 0])
        C = out[0] * (29.1066045841922 + 2.91058307855874) - 2.91058307855874
        D = out[1] * (4.8134735224616 + 3.97198033957307) - 3.97198033957307
        T1, T2 = 303.15, 363.15
        B = D * 2 * T1 * T2 / (T2 - T1); A = C - B * (T1 + T2) / (2 * T1 * T2)
        return A, B

def export(mod, args, names, outs, dyn, path):
    torch.onnx.export(mod, args, path, input_names=names, output_names=outs, dynamic_axes=dyn, opset_version=17, dynamo=False)

js, index, off = [], {}, 0
def put(name, a):
    global off
    a = np.ascontiguousarray(np.asarray(a, np.float32)); index[name] = [off, list(a.shape)]; js.append(a.ravel()); off += a.size
meta = {'seeds': SEEDS, 'dmpnn': {}, 'ghgnn': {}}
for k, s in enumerate(SEEDS):
    p = dict(np.load(PARAMS / f'dmpnn_{s}.npz')); info = json.loads((PARAMS / f'dmpnn_{s}.json').read_text())
    meta['dmpnn'][s] = {'scalers': info['scalers']}
    export(DPrep(p), (torch.randn(4, 256),), ['hs'], ['sp'], {'hs': {0: 'n'}, 'sp': {0: 'n'}}, ONX / f'dmpnn_prep_{k}.onnx')
    export(DHead(p, info['scalers']), (torch.randn(1, 256), torch.randn(4, 256), torch.randn(4, 512), torch.randn(2)), ['hi', 'hs', 'sp', 'tz'], ['lng'],
           {'hs': {0: 'n'}, 'sp': {0: 'n'}, 'tz': {0: 'K'}, 'lng': {0: 'n', 1: 'K'}}, ONX / f'dmpnn_head_{k}.onnx')
    for n in ('bond_input', 'atom_output', 'size_aware_readout'):
        put(f'd{k}/{n}/W', p[f'shared_encoder/{n}/kernel']); put(f'd{k}/{n}/b', p[f'shared_encoder/{n}/bias'])
    put(f'd{k}/bond_message/W', p['shared_encoder/bond_message/kernel'])
    m = GHGNN_model(M.n_atom_features(), M.n_bond_features(), 3, HP['hidden_dim']); m.load_state_dict(torch.load(MODELS / 'ghgnn' / s / 'final.pt', map_location='cpu')); m.eval()
    export(GPrep(m), (torch.randn(4, 226),), ['z'], ['p'], {'z': {0: 'n'}, 'p': {0: 'n'}}, ONX / f'ghgnn_prep_{k}.onnx')
    export(GHead(m), (torch.randn(4, 226), torch.randn(1, 226), torch.randn(226, 226)), ['p1', 'p2', 'Wv'], ['A', 'B'], {'p1': {0: 'n'}, 'A': {0: 'n'}, 'B': {0: 'n'}}, ONX / f'ghgnn_head_{k}.onnx')
    L = lambda mod: (mod.weight.detach().numpy().T, mod.bias.detach().numpy())
    for g in ('graphnet1', 'graphnet2'):
        gn = getattr(m, g)
        for part, seq in (('e', gn.edge_model.edge_mlp), ('n', gn.node_model.node_mlp), ('g', gn.global_model.global_mlp)):
            for j, li in ((1, 0), (2, 2)):
                Wt, b = L(seq[li]); put(f'g{k}/{g}/{part}{j}/W', Wt); put(f'g{k}/{g}/{part}{j}/b', b)
    for nm in ('gnorm1', 'gnorm2'):
        gnm = getattr(m, nm); put(f'g{k}/{nm}/w', gnm.weight.detach()); put(f'g{k}/{nm}/b', gnm.bias.detach()); put(f'g{k}/{nm}/ms', gnm.mean_scale.detach()); meta['ghgnn']['eps'] = float(gnm.eps)
    gc = m.global_conv1
    Wt, b = L(gc.gnn_layer.nn[0]); put(f'g{k}/enet1/W', Wt[0]); put(f'g{k}/enet1/b', b)
    Wt, b = L(gc.gnn_layer.nn[2]); put(f'g{k}/enet2/W', Wt); put(f'g{k}/enet2/b', b)
data = np.concatenate(js); data.astype('<f2').tofile(OUT / 'weights_js.f16'); data.astype('<f4').tofile(OUT / 'weights_js.f32')
meta['index'] = index; meta['n'] = int(off); meta['dtype'] = 'float16'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta['files'] = {str(p.relative_to(OUT)): sha(p) for p in sorted(list(ONX.glob('*.onnx')) + [OUT / 'weights_js.f16'])}
(OUT / 'weights_js.json').write_text(json.dumps(meta))
tot = sum((OUT / f).stat().st_size for f in meta['files']); print('model files MB', round(tot / 1e6, 1))
