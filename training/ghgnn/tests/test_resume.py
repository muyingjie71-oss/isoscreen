"""Resume-from-checkpoint test that runs without a GPU.

1. A 2-epoch smoke job writes last.pt.  2. The same job is started again and takes the resume branch.
On a GPU, torch.load(map_location=<cuda device>) also moves the saved CPU RNG states onto the GPU, and
torch.set_rng_state / Generator.set_state reject them ("RNG state must be a torch.ByteTensor").
Here torch.load is wrapped so that a map_location given as a torch.device object (the job passes its run
device, which is cuda on Colab) puts those RNG entries on a non-CPU device ('meta'), exactly reproducing
that failure; only the literal map_location='cpu' keeps them on the host. The resumed job must succeed and
leave model weights, optimiser state and RNG state unchanged (the smoke run has no epochs left to train)."""
import sys, json, runpy, shutil, tempfile
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
_load = torch.load
def _load_like_gpu(f, *a, map_location=None, **k):
    ck = _load(f, *a, map_location='cpu', **k)
    if isinstance(ck, dict) and 'torch_rng' in ck and isinstance(map_location, torch.device):
        for key in ('torch_rng', 'gen'):
            ck[key] = ck[key].to('meta')
    return ck

def run(args):
    sys.argv = ['run_ghgnn_job.py'] + args
    runpy.run_path(str(ROOT / 'run_ghgnn_job.py'), run_name='__main__')

def main():
    tmp = Path(tempfile.mkdtemp()); out = tmp / 'job'
    args = ['--data', str(ROOT / 'data'), '--seed', '1', '--fold', '1', '--out', str(out), '--smoke']
    run(args)
    fold = out / 'Fold_1'
    ck = _load(fold / 'last.pt', map_location='cpu', weights_only=False)
    (fold / 'smoke_result.json').unlink()
    torch.load = _load_like_gpu
    try: run(args)
    finally: torch.load = _load
    res = json.loads((fold / 'smoke_result.json').read_text())
    ck2 = _load(fold / 'last.pt', map_location='cpu', weights_only=False)
    same_model = all(torch.equal(ck['model'][k], ck2['model'][k]) for k in ck['model'])
    same_rng = torch.equal(ck['torch_rng'], torch.get_rng_state())
    assert res['epochs'] == ck['state']['epoch'] and same_model and same_rng, (res, same_model, same_rng)
    print(json.dumps({'status': 'PASS', 'resumed_at_epoch': res['epochs'], 'model_restored': same_model, 'torch_rng_restored': same_rng}))
    shutil.rmtree(tmp, ignore_errors=True)

if __name__ == '__main__':
    main()
