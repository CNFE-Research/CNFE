import secrets
from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from CNFE.lin import cipher as lin
from CNFE.utils.rng import create_rng

PublicParameter = namedtuple(
    "PublicParameter",
    [
        "l",  # int  # input size
        "m",  # int  # protocol parameter
        "n",  # int  # protocol parameter
        "p_1",  # int
        "p_2",  # int  # random parameter
        "alpha",  # float  # random parameter
        "lbda",  # int  # upper bound of the noise
    ],
)


@dataclass
class PK:
    lin: lin.PK
    u: np.array


@dataclass
class MSK:
    lin: lin.MSK


@dataclass
class CT:
    lin: lin.CT
    c: np.array


@dataclass
class SK:
    lin: lin.SK


class Simluator:
    param: PublicParameter
    rng = create_rng(secrets.randbits(256))

    def __init__(self, param: PublicParameter):
        self.param = param
        self.lin_param = lin.PublicParameter(
            l=param.l + 1,
            m=param.m,
            n=param.n,
            p_1=param.p_1,
            p_2=param.p_2,
            alpha=param.alpha,
            lbda=param.lbda,
        )
        self.lin_cipher = lin.Simluator(self.lin_param)

    def setup(self) -> Tuple[PK, MSK]:
        u = self.rng.integers(0, self.param.p_2, self.param.l)
        pk, msk = self.lin_cipher.setup()
        return PK(lin=pk, u=u), MSK(lin=msk)

    def enc(self, pk: PK, x: np.array):
        s = self.rng.integers(0, self.param.p_2)
        e = self.rng.normal(0, self.param.alpha, self.param.l).round(0).astype(int)

        c = s * pk.u + self.param.lbda * e + x

        lin_x = np.append(c * s, [s * s])

        lin_ct, r = self.lin_cipher.enc(pk.lin, lin_x)
        return CT(lin=lin_ct, c=c), r

    def key_gen(self, pk: PK, msk: MSK, a: np.matrix, r: int, sigma: int) -> SK:
        assert a.shape == (self.param.l, self.param.l)

        lin_y = np.zeros(self.lin_param.l)

        for i in range(self.param.l):
            for j in range(self.param.l):
                u_ij = np.zeros(self.lin_param.l)
                u_ij[i] = pk.u[i]
                u_ij[j] = pk.u[j]
                u_ij[self.param.l] = pk.u[i] * pk.u[j]

                lin_y += a[i][j] * u_ij

        return SK(lin=self.lin_cipher.key_gen(pk.lin, msk.lin, lin_y, r, sigma))

    def dec(self, a: np.matrix, sk: SK, ct: CT):
        res = 0
        for i in range(self.param.l):
            for j in range(self.param.l):
                res += a[i][j] * ct.c[i] * ct.c[j]

        return (res + self.lin_cipher.dec(sk.lin, ct.lin)) % self.param.lbda
