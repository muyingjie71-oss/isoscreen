"""Draw Figures 1-6 of the paper.

Usage (from this folder):
    python run_all.py            # all figures -> out/
    python run_all.py 3 6        # only Fig. 3 and Fig. 6
Each figure script can also be run alone:  python fig3.py . out
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'out'
which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6']
for n in which:
    print(f'--- Fig. {n}', flush=True)
    r = subprocess.run([sys.executable, str(ROOT / f'fig{n}.py'), str(ROOT), str(OUT)], cwd=ROOT)
    if r.returncode:
        sys.exit(f'Fig. {n} failed')
print('done ->', OUT)
