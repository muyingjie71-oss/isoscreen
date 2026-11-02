"""Device-consistency test that runs without a GPU.

The model runs on the CPU while torch.cuda.is_available() reports True. A TorchFunctionMode tracks a
"virtual device" for every tensor created inside forward/backward:
  * a factory call (torch.tensor, zeros, arange, ...) without device=  -> marked "host" (on a real GPU run it would sit on the CPU)
  * .cuda() / .to(<cuda>) / .to(device=<cuda>)                         -> unmarked (moved to the accelerator)
  * .type(torch.FloatTensor) / .cpu() / .to('cpu')                      -> marked "host"
  * any other op inherits "host" from its inputs
An op that mixes a "host" tensor with more than 0 dimensions and an accelerator tensor is exactly the case that raises
"Expected all tensors to be on the same device" on a GPU, so the test fails there. Covers ordinary batches and the
single-node (no-bond) dummy-graph path, train-mode forward+backward and eval-mode forward."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.overrides import TorchFunctionMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'ghgnn'))
torch.cuda.is_available = lambda: True          # take the GPU branches of the upstream code
import run_ghgnn_job as J
import mol2graph as M
from GHGNN_architecture import GHGNN_model

FACTORIES = {torch.tensor, torch.as_tensor, torch.from_numpy, torch.zeros, torch.ones, torch.empty, torch.full,
             torch.arange, torch.linspace, torch.eye, torch.rand, torch.randn, torch.randint}
# In this emulation a torch.device object stands for the accelerator: upstream builds them as
# torch.device('cuda' if available else 'cpu') (-> cuda here), and the adapter takes them from tensor.device,
# which on a real GPU run is cuda. Only the literal string 'cpu' denotes the host.
def _is_cuda(x):
    return isinstance(x, torch.device) or (isinstance(x, str) and x.startswith('cuda'))
def _is_cpu(x):
    return isinstance(x, str) and x == 'cpu'
def _tensors(o):
    if isinstance(o, torch.Tensor): yield o
    elif isinstance(o, (list, tuple)):
        for v in o: yield from _tensors(v)
    elif isinstance(o, dict):
        for v in o.values(): yield from _tensors(v)
def _host(t): return getattr(t, '_virtual_host', False)
def _mark(o, v):
    for t in _tensors(o):
        try: t._virtual_host = v
        except Exception: pass
    return o

class DeviceMismatch(RuntimeError): pass

class VirtualDevice(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, '__name__', '')
        if func in FACTORIES:
            dev = kwargs.get('device')
            if dev is not None: kwargs = {**kwargs, 'device': 'cpu'}
            return _mark(func(*args, **kwargs), not _is_cuda(dev))
        if func is torch.Tensor.cuda:
            return _mark(args[0].clone(), False)
        if func is torch.Tensor.cpu:
            return _mark(func(*args, **kwargs), True)
        if func is torch.Tensor.to:
            targets = list(args[1:]) + [kwargs.get('device')]
            to_cuda = any(_is_cuda(x) for x in targets); to_cpu = any(_is_cpu(x) for x in targets)
            args = tuple('cpu' if _is_cuda(x) else x for x in args)
            if _is_cuda(kwargs.get('device')): kwargs = {**kwargs, 'device': 'cpu'}
            out = func(*args, **kwargs)
            if to_cuda: return _mark(out.clone() if out is args[0] else out, False)
            if to_cpu: return _mark(out.clone() if out is args[0] else out, True)
            return _mark(out, _host(args[0]))
        if func is torch.Tensor.type and len(args) > 1 and args[1] in (torch.FloatTensor, torch.LongTensor, 'torch.FloatTensor', 'torch.LongTensor'):
            return _mark(func(*args, **kwargs).clone(), True)
        ins = [t for t in _tensors((args, kwargs))]
        host = [t for t in ins if _host(t) and t.dim() > 0]
        acc = [t for t in ins if not _host(t)]
        if host and acc and name not in ('__get__', '__set__', 'requires_grad_', 'retain_grad', 'register_hook'):
            raise DeviceMismatch(f'{name}: mixes a host-side tensor {tuple(host[0].shape)} with accelerator tensors '
                                 '(on a GPU this raises "Expected all tensors to be on the same device")')
        out = func(*args, **kwargs)
        if any(_host(t) for t in ins) and not acc: _mark(out, True)
        elif any(_host(t) for t in ins):
            _mark(out, False)  # only 0-dim host scalars mixed in: PyTorch puts the result on the accelerator
        return out

def main(data):
    D = Path(data)
    t = pd.read_csv(D / 'task1.csv', keep_default_na=False)
    mol = pd.read_csv(D / 'molecules.csv'); smi = dict(zip(mol.Mol_ID, mol.SMILES))
    from rdkit import Chem
    mols = {k: Chem.MolFromSmiles(s) for k, s in smi.items()}
    ok = {k for k, m in mols.items() if all(a.GetSymbol() not in ('Al', 'Co', 'Fe', 'Sb') for a in m.GetAtoms())}
    t = t[t.Solvent_Mol_ID.isin(ok) & t.Solute_Mol_ID.isin(ok)]
    nb = {k: m.GetNumBonds() for k, m in mols.items()}
    single = t[t.Solvent_Mol_ID.map(nb).eq(0) | t.Solute_Mol_ID.map(nb).eq(0)]
    rng = np.random.default_rng(0)
    parts = {'ordinary': t.iloc[rng.choice(len(t), 64, replace=False)]}
    if len(single): parts['single_node'] = single.iloc[:8]
    torch.manual_seed(0)
    model = GHGNN_model(M.n_atom_features(), M.n_bond_features(), 3, 113)
    dev = torch.device('cuda')
    res = {}
    for name, df in parts.items():
        ids = set(df.Solvent_Mol_ID) | set(df.Solute_Mol_ID)
        C = J.mol_cache({k: smi[k] for k in ids})
        rows = list(zip(df.Solvent_Mol_ID, df.Solute_Mol_ID, df.ln_gamma_inf.astype(float), df.T_K.astype(float)))
        nbatch = 0
        for s, u, T in J.loader(J.build(rows, C), 32, False):
            s0, u0, T0 = s.clone(), u.clone(), T.clone()   # upstream forward may modify a batch in place: fresh copy per forward
            with VirtualDevice():
                s, u, T = s.to(dev), u.to(dev), T.to(dev)
                y = s.y.float().clone()       # same order as run_ghgnn_job.py: labels read before forward
                model.train(); pred = model(s, u, T)
                loss = ((pred.float() - y.reshape(pred.shape)) ** 2).mean()
                loss.backward()
                model.eval()
                s, u, T = s0.to(dev), u0.to(dev), T0.to(dev)
                with torch.no_grad(): pe = model(s, u, T)
                assert pe.shape[0] == s0.num_graphs, (pe.shape, s0.num_graphs)
            assert torch.isfinite(pred).all()
            nbatch += 1
        res[name] = {'records': len(df), 'batches': nbatch}
    print(json.dumps({'status': 'PASS', 'check': 'virtual-device consistency (GPU code paths on CPU)', 'checked': res}))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'data'))
