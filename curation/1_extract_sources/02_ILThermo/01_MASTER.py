"""Build the master table (01_MASTER) of this source."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from _pipeline.cli import run_entry
if __name__=="__main__":
    run_entry(Path(__file__),"01_MASTER")
