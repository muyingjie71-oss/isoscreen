"""One workbook per public entrypoint, without implicit downstream writes."""
from pathlib import Path
import argparse,json,hashlib
from .context import SourceContext
from .science import build_source_table

def run_entry(entry,kind):
    from .runtime import ensure_scientific_runtime
    ensure_scientific_runtime()
    ap=argparse.ArgumentParser(description='Rebuild only '+kind+'.xlsx from this source and reviewed identity evidence.')
    ap.add_argument('--work-dir',type=Path,help='Staging/verification directory; default is the D-drive working area.')
    ap.add_argument('--project-root',type=Path,help='Explicit raw-source project root for staging verification or a relocated project.')
    ap.add_argument('--selection',type=Path,help='A reviewed strict-identity decision manifest. No online guessing is permitted.')
    ap.add_argument('--verify-only',action='store_true',help='Build and validate the scientific bundle without replacing the workbook.')
    args=ap.parse_args();ctx=SourceContext.from_entry(entry,args.work_dir,args.project_root);target=ctx.output_path(kind)
    selection=args.selection
    if kind!='01_MASTER'and selection is None:
        folder=ctx.pipeline_root/'assets'/'identity_selection'
        lock=folder/'strict_identity_manifest_v3_final.json'
        if not lock.is_file():raise FileNotFoundError('Locked reviewed identity manifest missing; no output was changed: '+str(lock))
        manifest=json.loads(lock.read_text(encoding='utf8'));binding=manifest['decision_jsonl_by_source'][ctx.source_folder]
        renamed=folder/('strict_identity_'+ctx.source_folder+'.jsonl.gz')
        selection=renamed if renamed.is_file()else folder/binding['filename']
        if not selection.is_file():raise FileNotFoundError('Reviewed strict identity decisions missing; no output was changed: '+str(selection))
        with selection.open('rb')as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=binding['sha256']:raise ValueError('Reviewed identity asset SHA changed; no output was changed: '+str(selection))
    bundle=build_source_table(ctx,kind,selection)
    print(json.dumps({'status':'SCIENTIFIC_BUNDLE_VERIFIED','bundle':str(bundle)},ensure_ascii=False))
