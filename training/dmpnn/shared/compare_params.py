"""Bitwise comparison of model parameters in two checkpoints (timing fields in history excluded)."""
import sys,hashlib,numpy as np
from flax import serialization
def params_sha(path):
    s=serialization.msgpack_restore(open(path,'rb').read())['state']['params']
    h=hashlib.sha256()
    def walk(x,prefix=''):
        if isinstance(x,dict):
            for k in sorted(x):walk(x[k],prefix+'/'+k)
        else:h.update(prefix.encode());h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    walk(s);return h.hexdigest()
a,b=params_sha(sys.argv[1]),params_sha(sys.argv[2])
print('params sha',a[:16],b[:16],'IDENTICAL' if a==b else 'DIFFERENT')
sys.exit(0 if a==b else 3)
