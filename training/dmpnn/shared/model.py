"""Directed message-passing neural network (D-MPNN) with one output, ln gamma_inf."""
import jax
import jax.numpy as jnp
from flax import linen as nn

class DMPNN(nn.Module):
    hidden:int=256
    depth:int=4
    @nn.compact
    def __call__(self,g):
        nm=g['node_mask'][:,None];em=g['edge_mask'][:,None]
        a=g['atom']*nm
        h0=nn.silu(nn.Dense(self.hidden,name='bond_input')(jnp.concatenate([a[g['src']],g['bond']],axis=-1)))*em
        h=h0
        update=nn.Dense(self.hidden,use_bias=False,name='bond_message')
        for _ in range(self.depth-1):
            incoming=jax.ops.segment_sum(h,g['dst'],num_segments=a.shape[0])
            msg=incoming[g['src']]-h[g['rev']]  # exclude reverse bond
            h=nn.silu(h0+update(msg))*em
        incoming=jax.ops.segment_sum(h,g['dst'],num_segments=a.shape[0])
        nodes=nn.silu(nn.Dense(self.hidden,name='atom_output')(jnp.concatenate([a,incoming],axis=-1)))*nm
        ngraph=g['graph_mask'].shape[0]
        pooled=jax.ops.segment_sum(nodes,g['node_graph'],num_segments=ngraph)
        count=jax.ops.segment_sum(nm,g['node_graph'],num_segments=ngraph)
        # Original mean-only pooling made e.g. same-element rings of different
        # sizes indistinguishable. Fuse mean AND sum AND explicit size; keep
        # masked padded graphs precisely zero.
        mean=pooled/jnp.maximum(count,1)
        features=jnp.concatenate([mean,pooled,jnp.log1p(count)],axis=-1)
        h=nn.silu(nn.Dense(self.hidden,name='size_aware_readout')(features))
        return h*g['graph_mask'][:,None]

class IDACModel(nn.Module):
    hidden:int=256
    depth:int=4
    dropout:float=.10
    @nn.compact
    def __call__(self,b,training=False):
        h=DMPNN(self.hidden,self.depth,name='shared_encoder')(b['graph'])
        hi,hs=h[b['solute_graph']],h[b['solvent_graph']]
        ht=nn.silu(nn.Dense(32,name='temperature')(b['temperature'][:,None]))
        n=hi.shape[0]
        zero=jnp.zeros((n,128),dtype=jnp.float32)
        # Fusion input in solute-solvent order. The three blocks of zeros are part of the input
        # layout of the trained models and carry no information.
        x=jnp.concatenate([hi,hs,hi-hs,hi*hs,ht,zero,zero,zero],axis=-1)
        for nout in (512,256):
            x=nn.silu(nn.Dense(nout,name=f'fusion_{nout}')(x))
            x=nn.Dropout(rate=self.dropout)(x,deterministic=not training)
        return nn.Dense(1,name='ln_gamma')(x).squeeze(-1).astype(jnp.float32)*b['record_mask']
