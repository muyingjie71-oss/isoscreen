import jax.numpy as jnp
import optax
def masked_mean(x,mask):return jnp.sum(x*mask)/jnp.maximum(jnp.sum(mask),1)
def joint_loss(pred,b,lam=1.,delta=1.):
    abs_loss=masked_mean(optax.huber_loss(pred[b['absolute_map']],b['absolute_y'],delta=delta),b['absolute_mask'])
    diff=pred[b['pair_a_map']]-pred[b['pair_b_map']]
    # Groups sampled uniformly, one uniformly selected pair per sampled group.
    # This directly estimates mean_g mean_pair Huber; do not reweight here.
    diff_loss=masked_mean(optax.huber_loss(diff,b['pair_y'],delta=delta),b['pair_mask'])
    return abs_loss+lam*diff_loss,{'absolute_loss':abs_loss,'differential_loss':diff_loss}
