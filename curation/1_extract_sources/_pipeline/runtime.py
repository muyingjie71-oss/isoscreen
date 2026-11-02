"""Use the installed D-drive scientific libraries without installing anything."""
from pathlib import Path
import os,sys,importlib.util

def ensure_scientific_runtime():
    configured=os.environ.get('SURROGATE_SCIENCE_SITE_PACKAGES','')
    candidates=[Path(p)for p in configured.split(os.pathsep)if p]
    candidates+=[Path('work/ml_env/Lib/site-packages'),Path('work/source_unification_v1/runtime_deps')]
    added=[]
    for p in candidates:
        if p.is_dir()and str(p)not in sys.path:sys.path.insert(0,str(p));added.append(str(p))
    if importlib.util.find_spec('rdkit')is None:
        raise RuntimeError('缺少科学结构运行库rdkit。请使用已配置的D盘环境，或按_pipeline/dependency_requirements.txt安装；尚未更改Excel。')
    return {'extra_scientific_library_paths':added,'automatic_installation':False}
