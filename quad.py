from CNFE.quad.cipher import PublicParameter, Simluator
from CNFE.utils.rng import create_rng
import math
import numpy as np

rng = create_rng()

sigma = 10

param = PublicParameter(
    l=64,  # input size
    m=4,  # protocol parameter
    n=4,  # protocol parameter
    p_1=10**5,  # upper bound of the inner product
    p_2=10**6,  # random parameter
    alpha=10**5,  # random parameter
    lbda=9 * 10**5,  # upper bound of the noise
)


x = rng.integers(0, math.floor(math.sqrt(param.p_1 / param.l)), size=param.l)

a = np.ones((param.l, param.l))

sim = Simluator(param)

pk, msk = sim.setup()
ct, r = sim.enc(pk, x)
sk = sim.key_gen(pk, msk, a, r, sigma)
result = sim.dec(a, sk, ct)

print(result)
