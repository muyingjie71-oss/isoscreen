"""Drop-in replacements for torch_scatter.scatter_add / scatter_mean (dim=0), built on torch.index_add_.
Semantics match torch_scatter: empty segments give 0; `out` (if given) fixes the output shape."""
import torch
def scatter_add(src,index,dim=0,dim_size=None,out=None):
    assert dim==0
    n=out.shape[0] if out is not None else (dim_size if dim_size is not None else int(index.max())+1 if index.numel() else 0)
    res=torch.zeros((n,)+tuple(src.shape[1:]),dtype=src.dtype,device=src.device)
    return res.index_add_(0,index,src)
def scatter_mean(src,index,dim=0,dim_size=None,out=None):
    s=scatter_add(src,index,dim,dim_size,out)
    cnt=torch.zeros(s.shape[0],dtype=src.dtype,device=src.device).index_add_(0,index,torch.ones(index.shape[0],dtype=src.dtype,device=src.device))
    return s/cnt.clamp(min=1).view(-1,*([1]*(s.dim()-1)))
