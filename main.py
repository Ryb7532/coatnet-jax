import jax
import jax.numpy as jnp

from flax.training import train_state
from clu import parameter_overview

from coatnet import (
    coatnet_0,
    coatnet_1,
    coatnet_2,
    coatnet_3,
    coatnet_4,
    coatnet_5,
    coatnet_6,
    coatnet_7,
)

def main():
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 224, 224, 3))

    model = coatnet_0(1000)
    params = model.init(key, x)
    out = model.apply(params, x, train=False)
    print(out.shape, sum(x.size for x in jax.tree_util.tree_leaves(params)))

    model = coatnet_1(1000)
    params = model.init(key, x)
    out = model.apply(params, x, train=False)
    print(out.shape, sum(x.size for x in jax.tree_util.tree_leaves(params)))

    model = coatnet_2(1000)
    params = model.init(key, x)
    out = model.apply(params, x, train=False)
    print(out.shape, sum(x.size for x in jax.tree_util.tree_leaves(params)))

    model = coatnet_3(1000)
    params = model.init(key, x)
    out = model.apply(params, x, train=False)
    print(out.shape, sum(x.size for x in jax.tree_util.tree_leaves(params)))

    model = coatnet_4(1000)
    params = model.init(key, x)
    out = model.apply(params, x, train=False)
    print(out.shape, sum(x.size for x in jax.tree_util.tree_leaves(params)))

    model = coatnet_5(1000)
    params = model.init(key, x)
    out = model.apply(params, x, train=False)
    print(out.shape, sum(x.size for x in jax.tree_util.tree_leaves(params)))

    model = coatnet_6(1000)
    params = model.init(key, x)
    out = model.apply(params, x, train=False)
    print(out.shape, sum(x.size for x in jax.tree_util.tree_leaves(params)))

    model = coatnet_7(1000)
    params = model.init(key, x)
    out = model.apply(params, x, train=False)
    print(out.shape, sum(x.size for x in jax.tree_util.tree_leaves(params)))


if __name__ == '__main__':
    main()
