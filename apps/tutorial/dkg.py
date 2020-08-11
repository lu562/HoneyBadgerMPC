import asyncio
import logging
from charm.toolbox.ecgroup import ECGroup, G, ZR
from charm.toolbox.eccurve import secp256k1
from honeybadgermpc.utils.misc import subscribe_recv, wrap_send
from honeybadgermpc.polynomial import EvalPoint, polynomials_over
from honeybadgermpc.preprocessing import (
    PreProcessedElements as FakePreProcessedElements,
)
from honeybadgermpc.elliptic_curve import Subgroup
from honeybadgermpc.progs.mixins.share_arithmetic import (
    BeaverMultiply,
    BeaverMultiplyArrays,
    MixinConstants,
)
from honeybadgermpc.progs.mixins.dataflow import ShareArray
from honeybadgermpc.field import GF
import honeybadgermpc.progs.fixedpoint 
import time
import random
from collections import defaultdict
from honeybadgermpc.utils.misc import subscribe_recv, wrap_send
mpc_config = {
    MixinConstants.MultiplyShareArray: BeaverMultiplyArrays(),
    MixinConstants.MultiplyShare: BeaverMultiply(),
}
# public parameters
p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
Field = GF(p)
group256 = ECGroup(secp256k1)
g = group256.init(G, 999999)
h = group256.init(G, 20)


# excptions
class FieldsNotIdentical(Exception):
    pass
class DegreeNotIdentical(Exception):
    pass
class ShareNotValid(Exception):
    pass

# Pederson commitments
def Pederson_commit(g, h, coefficients, h_coefficients):

    decoded_commitments = [0 for _ in range(len(coefficients))]
    for i in range(len(coefficients)):
        decoded_commitments[i] = group256.serialize(g ** group256.init(ZR, int(coefficients[i].value)) * h ** group256.init(ZR, int(h_coefficients[i].value)))

    return decoded_commitments

def Pederson_verify(g, h, x, share, serialized_commitments):
    left_hand_side = g ** group256.init(ZR, share[0].value) * h ** group256.init(ZR, share[1].value)
    right_hand_side = group256.init(G, int(1))
    commitments = [0 for _ in range(len(serialized_commitments))]
    v = 0
    for i in range(len(serialized_commitments)):
        commitments[i] =group256.deserialize(serialized_commitments[i])

    for i in range(len(commitments)):
        e = group256.init(ZR, int(x ** i))
        right_hand_side = right_hand_side * (commitments[i] ** e)

    if left_hand_side == right_hand_side:
        return True
    else:
        return False

class VSSShare:
    def __init__(self, ctx, value, value_prime, commitments, valid):
        self.ctx = ctx
        self.value = value
        self.value_prime = value_prime
        self.commitments = commitments
        self.valid = valid

    def __add__(self, other):
        """Addition."""
        if not isinstance(other, (VSSShare)):
            return NotImplemented
        if not self.valid or not other.valid:
            raise ShareNotValid
        if self.value.field is not other.value.field:
            raise FieldsNotIdentical
        if len(self.commitments) != len(other.commitments):
            raise DegreeNotIdentical
        sum_commitment = [group256.serialize(group256.deserialize(self.commitments[i]) * group256.deserialize(self.commitments[i])) for i in range(len(self.commitments))]
        return VSSShare(self.ctx, self.value + other.value,self.value_prime + other.value_prime, sum_commitment, True)
    async def open(self):
        res = self.ctx.GFElementFuture()
        temp_share = self.ctx.preproc.get_zero(self.ctx) + self.ctx.Share(int(self.value.value))
        opened_value = await temp_share.open()

        return opened_value

class VSS:
    def __init__(self, ctx,field = Field, g=g, h=h):
        self.ctx = ctx
        self.field = field
        self.send = ctx.send
        self.recv = ctx.recv
        self.g = g
        self.h = h
        self.N = ctx.N
        self.t = ctx.t
        self.vss_id = 0
        self.my_id = ctx.myid
        self.poly = polynomials_over(self.field)

    def _get_share_id(self):
        """Returns a monotonically increasing int value
        each time this is called
        """
        share_id = self.vss_id
        self.vss_id += 1
        return share_id

    async def share(self, dealer_id, value):
        if type(value) is int:
            value = Field(value)
        shareid = self._get_share_id()
        # Share phase of dealer
        if dealer_id == self.my_id:
            # generate polynomials
            poly_f = self.poly.random(self.t, value)
            poly_f_prime = self.poly.random(self.t, Field.random())

            commitments = Pederson_commit(self.g, self.h, poly_f.coeffs, poly_f_prime.coeffs)
            messages = [0 for _ in range(self.N)]
            # send f(1) to party 0. we cannot send f(0) as it is the secret
            for i in range(self.N):
                messages[i] = [poly_f(i + 1),poly_f_prime(i + 1)]


            for dest in range(self.N):
                self.send(dest, ("VSS", shareid, [commitments, messages[dest]]))

        # Share phase of recipient parties(including dealer)        
        share_buffer = self.ctx._vss_buffers[shareid] 
        msg, _ = await asyncio.wait([share_buffer], return_when=asyncio.ALL_COMPLETED)
        # there is only one element in msg, but I don't know other way to traverse a set
        for i in msg:  
            commitments = i.result()[0]
            share = i.result()[1]
            valid = Pederson_verify(self.g, self.h, self.my_id + 1, share, commitments)
            
            return VSSShare(self.ctx, share[0], share[1], commitments, valid)







async def run(ctx, **kwargs):
    V = VSS(ctx)
    # to send a vss share, use "await V.share(dealer, value)"
    a = await V.share(0, 30)
    b = await V.share(0, 40)
    c = a + b
    open_c = await c.open()
    print(open_c)


async def _run(peers, n, t, my_id, k):
    from honeybadgermpc.ipc import ProcessProgramRunner

    async with ProcessProgramRunner(peers, n, t, my_id, mpc_config) as runner:
        await runner.execute("0", run, k=k)
        bytes_sent = runner.node_communicator.bytes_sent
        print(f"[{my_id}] Total bytes sent out: {bytes_sent}")


if __name__ == "__main__":
    from honeybadgermpc.config import HbmpcConfig
    import sys

    HbmpcConfig.load_config()

    if not HbmpcConfig.peers:
        print(
            f"WARNING: the $CONFIG_PATH environment variable wasn't set. "
            f"Please run this file with `scripts/launch-tmuxlocal.sh "
            f"apps/tutorial/hbmpc-tutorial-2.py conf/mpc/local`"
        )
        sys.exit(1)

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    k = 1000
    try:
        pp_elements = FakePreProcessedElements()
        if HbmpcConfig.my_id == 0:
            
            pp_elements.generate_zeros(20, HbmpcConfig.N, HbmpcConfig.t)
            # pp_elements.generate_triples(2, HbmpcConfig.N, HbmpcConfig.t)
            # pp_elements.generate_bits(20, H0bmpcConfig.N, HbmpcConfig.t)
            pp_elements.preprocessing_done()
        else:
            loop.run_until_complete(pp_elements.wait_for_preprocessing())

        loop.run_until_complete(
            _run(HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id, k)
        )
    finally:
        loop.close()
