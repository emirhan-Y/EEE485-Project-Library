from max_pooling_layer import max_pooling_layer

import numpy as np

X = np.arange(1, 101).reshape(10, 10)
pool_dim = (3, 3)
pool = max_pooling_layer(X.shape, pool_dim)
Y = pool.fwd_prop(X)

dE_dY = np.ones_like(Y)
dE_dX = pool.bck_prop(dE_dY, 1)
print('foo')
