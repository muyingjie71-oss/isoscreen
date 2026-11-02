from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from _pipeline.curation_entry import run_curation_entry
if __name__=="__main__":
    run_curation_entry(__file__,"01_MERGED_DEDUPLICATED")
