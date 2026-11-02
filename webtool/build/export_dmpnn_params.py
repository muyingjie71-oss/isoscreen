"""Flatten the production D-MPNN flax params (msgpack) to npz, one file per seed. Run with the JAX env."""
import sys, json, numpy as np
from pathlib import Path
from flax import serialization
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MODELS, PARAMS
src = Path(sys.argv[1]) if len(sys.argv) > 1 else MODELS / 'dmpnn'
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else PARAMS; dst.mkdir(parents=True, exist_ok=True)
for d in sorted(src.glob('seed_*')):
    p = serialization.msgpack_restore((d/'final_params.msgpack').read_bytes()); flat = {}
    def walk(x, pre):
        if isinstance(x, dict):
            for k, v in x.items(): walk(v, pre + [k])
        else: flat['/'.join(pre)] = np.asarray(x, np.float32)
    walk(p, [])
    np.savez(dst / f'dmpnn_{d.name}.npz', **flat)
    (dst / f'dmpnn_{d.name}.json').write_text(json.dumps({'config': json.loads((d/'config.json').read_text()), 'scalers': json.loads((d/'scalers.json').read_text())}))
    print(d.name, {k: v.shape for k, v in flat.items()})
