from __future__ import annotations
from pathlib import Path
import hashlib,json,itertools

def json_value(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,set):return sorted(value)
    raise TypeError(type(value).__name__)

class RowStore:
    """Repeatable on-disk records; large MASTER tables never become a list."""
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self._out=self.path.open('w',encoding='utf8',newline='\n');self.count=0
    def append(self,row):
        self._out.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),default=json_value)+'\n');self.count+=1
    write_row=append
    def close(self):
        if not self._out.closed:self._out.close()
    def __iter__(self):
        self.close()
        with self.path.open(encoding='utf8')as f:
            for line in f:
                if line.strip():yield json.loads(line)
    @property
    def sha256(self):
        self.close()
        with self.path.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def column(label):
    kind='number'if any(k in label for k in ('温度_K','压力_kPa','无限稀释活度系数','活度系数不确定度','行数','记录数','成员数'))else'text'
    width=38 if 'InChIKey'in label else 64 if 'InChI'in label or 'SMILES'in label else 65 if any(k in label for k in ('原因','依据','质量备注','位置'))else 34 if label in ('溶质','溶剂','当前异构体名称')else 18
    return {'label':label,'width':width,'kind':kind,'hidden':False}

def materialize_sheet(directory,name,headers,rows,header_rows=1,kind='table',sections=None,freeze_columns=0):
    file=Path(directory)/(name+'.jsonl');file.parent.mkdir(parents=True,exist_ok=True);count=0;samples=[]
    with file.open('w',encoding='utf8',newline='\n')as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),default=json_value)+'\n')
            if count<3:samples.append(row)
            count+=1
    return {'name':name,'columns':[column(h)for h in headers],'file':str(file.resolve()),'rows':count,'samples':samples,'header_rows':header_rows,'kind':kind,'sections':sections or [],'freeze_columns':freeze_columns}

def materialize_split_table(directory,prefix,headers,rows,max_rows=1000000,**kwargs):
    iterator=iter(rows);sheets=[];sentinel=object();n=0
    while True:
        first=next(iterator,sentinel)
        if first is sentinel:
            if not sheets:sheets.append(materialize_sheet(directory,prefix+'_001',headers,[],**kwargs))
            break
        n+=1;chunk=itertools.chain([first],itertools.islice(iterator,max_rows-1))
        sheets.append(materialize_sheet(directory,prefix+'_'+str(n).zfill(3),headers,chunk,**kwargs))
    return sheets

def save_manifest(ctx,kind,directory,statistics,flow,sheets,extra=None):
    manifest={'source_folder':ctx.source_folder,'source_name':ctx.source_name,'kind':kind,'statistics':statistics,'flow':flow,'sheets':sheets,**(extra or {})}
    path=Path(directory)/'bundle.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=json_value),encoding='utf8');return path
