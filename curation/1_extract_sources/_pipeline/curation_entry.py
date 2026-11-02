"""Each public entry writes exactly its matching curation workbook."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,os

NAMES=('01_MERGED_DEDUPLICATED','02_EXPERIMENTS_PAIRS_ISOMERS')

def run_curation_entry(entry,kind):
 from .runtime import ensure_scientific_runtime
 ensure_scientific_runtime()
 if kind not in NAMES:raise ValueError('Unsupported curation output')
 ap=argparse.ArgumentParser(description='从七份已审核03实验重建且只写入本脚本对应的一张Excel。')
 ap.add_argument('--project-root',type=Path)
 ap.add_argument('--work-dir',type=Path,help='构建与核验临时目录；默认 work/curation_rebuild。')
 ap.add_argument('--output',type=Path,help='明确输出位置；不指定时更新本脚本同名Excel。')
 ap.add_argument('--verify-only',action='store_true',help='只构建和检查科学描述，不写Excel。')
 args=ap.parse_args();entry=Path(entry).resolve();project=(args.project_root or entry.parent.parent).resolve()
 work=(args.work_dir or Path(os.environ.get('SURROGATE_CURATION_WORK','work/curation_rebuild'))).resolve()
 if project==work or project in work.parents:raise ValueError('临时工作目录必须放在数据项目外。')
 work.mkdir(parents=True,exist_ok=True)
 from .curation_science import build_current_science
 bank=build_current_science(project,work/'science')
 if kind==NAMES[0]:
  from .curation_book1_builder import build_book1_descriptor
  bundle=build_book1_descriptor(bank,work/kind)
 else:
  from .curation_book2_builder import build_book2_descriptor
  bundle=build_book2_descriptor(bank,work/kind)
 print(json.dumps({'status':'VERIFIED','bundle':str(bundle)},ensure_ascii=False))
