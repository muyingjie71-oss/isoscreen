from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

SOURCE_MODULES={'01_ThermoML':'thermoml','02_ILThermo':'ilthermo','03_Brouwer':'brouwer','04_DeepFM':'deepfm','05_Tan':'tan','06_Sherman':'sherman','07_单独收集论文':'papers'}
SOURCE_NAMES={'01_ThermoML':'ThermoML','02_ILThermo':'ILThermo','03_Brouwer':'Brouwer','04_DeepFM':'DeepFM','05_Tan':'Tan','06_Sherman':'Sherman','07_单独收集论文':'单独收集论文'}

@dataclass(frozen=True)
class SourceContext:
    source_folder:str
    project_root:Path
    work_dir:Path
    pipeline_root:Path=Path(__file__).resolve().parent
    @property
    def source_dir(self):return self.project_root/self.source_folder
    @property
    def raw_dir(self):return self.source_dir/'00_Source'
    @property
    def asset_dir(self):
        central=self.pipeline_root/'assets'/'sources'/self.source_folder
        return central if central.exists()else self.source_dir
    @property
    def source_name(self):return SOURCE_NAMES[self.source_folder]
    @property
    def module_name(self):return SOURCE_MODULES[self.source_folder]
    def output_path(self,kind):
        if kind not in ('01_MASTER','02_IDAC','03_ISOMERS'):raise ValueError('Unknown table kind')
        if kind=='01_MASTER'and self.source_folder not in ('01_ThermoML','02_ILThermo'):raise ValueError('This source has no MASTER table')
        return self.source_dir/(kind+'.xlsx')
    @classmethod
    def from_entry(cls,entry,work_dir=None,project_root=None):
        source_dir=Path(entry).resolve().parent
        if source_dir.name not in SOURCE_MODULES:raise ValueError('Entrypoint must be in a recognized source folder')
        default=os.environ.get('SURROGATE_WORK_DIR','work/source_unification_v1/runtime')
        return cls(source_dir.name,Path(project_root).resolve()if project_root else source_dir.parent,Path(work_dir or default).resolve())
