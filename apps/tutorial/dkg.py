import asyncio
import logging
from charm.toolbox.ecgroup import ECGroup, G, ZR
from charm.toolbox.eccurve import secp256k1, sect571k1
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
# sect571k1 = 0x020000000000000000000000000000000000000000000000000000000000000000000000131850E1F19A63E4B391A8DB917F4138B630D84BE5D639381E91DEB45CFE778F637C1001
# secp256k1 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
p = 0x020000000000000000000000000000000000000000000000000000000000000000000000131850E1F19A63E4B391A8DB917F4138B630D84BE5D639381E91DEB45CFE778F637C1001
Field = GF(p)
group256 = ECGroup(sect571k1)
g = group256.init(G, 999999)
h = group256.init(G, 20)
KAPPA = 64 # security parameter
K = 256 # bit length of integer
random.seed(2020)
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
        sum_commitment = [group256.serialize(group256.deserialize(self.commitments[i]) * group256.deserialize(other.commitments[i])) for i in range(len(self.commitments))]
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


#######################################    Bit Decomposation        ##############################
# MPC operations for fixed point
async def random2m(ctx, m):
    result = ctx.Share(0)
    bits = []
    for i in range(m):
        bits.append(ctx.preproc.get_bit(ctx))
        result = result + Field(2) ** i * bits[-1]

    return result, bits

async def Rand_mod2(ctx):
    result = ctx.Share(0)
    value, bits = await random2m(ctx, m=K)
    value = (value - bits[0]) * (1/Field(2))

    return value, bits[0]

async def Mod2(ctx, x, k = K):
    r, r_b = await Rand_mod2(ctx)
    c = await (x + Field(2) * r + r_b).open()
    c0 = Field(c.value % 2)
    result = c0 + r_b - 2 * c0 * r_b
    return result

### This is fake offline phase.
async def offline_PreMul(ctx, n):
    r = []
    s = []
    for _ in range(n):
        r.append(ctx.preproc.get_rand(ctx))
        s.append(ctx.preproc.get_rand(ctx))
    r_array = ctx.ShareArray(r)
    s_array = ctx.ShareArray(s)
    rs = await (r_array * s_array)
    rs_open = await rs.open()
    return r, s, rs_open


async def PreMul(ctx, shares):
    w = [0 for _ in range(len(shares))]
    z = [0 for _ in range(len(shares))]
    result = [0 for _ in range(len(shares))]
    r, s, u = await offline_PreMul(ctx,len(shares))
    v = await (ctx.ShareArray(r[1:]) * ctx.ShareArray(s[: -1]))
    w[0] = r[0]
    for i in range(1, len(shares)):
        w[i] = v._shares[i - 1] * (1/u[i-1])
    for i in range(len(shares)):
        z[i] = s[i] * (1/u[i])
    m = await (ctx.ShareArray(w) * ctx.ShareArray(shares))
    m_open = await m.open()
    result[0] = shares[0]
    temp = m_open[0]
    for i in range(1, len(shares)):
        temp = temp * m_open[i]
        result[i] = z[i] * temp

    return result

async def PreAnd(ctx, shares):
    result = await PreMul(ctx, shares)
    return result

# input e is in the form of [(s_0,s_1,s_n)(p_0,p_1,...,p_n))(k_0....k_n)]
async def carry_prop(ctx, e):

    p_delta = await PreAnd(ctx, e[1][::-1])
    p_delta = p_delta[::-1]
    k_p_delta = await (ctx.ShareArray(e[2][:-1]) * ctx.ShareArray(p_delta[1:]))
    c = e[2][-1]
    for i in range(len(k_p_delta._shares)):
        c = c + k_p_delta._shares[i]
    b = p_delta[0]
    a = Field(1) - b - c
    return [a,b,c]

########## parallelize carry_prop  ################

async def batch_carry_prop(ctx, es, offline):
    p_delta = await batch_PreAnd(ctx, [es[i][1][::-1] for i in range(len(es))], offline)
    p_delta = [p_delta[i][::-1] for i in range(len(p_delta))]
    temp_a = []
    temp_b = []
    for i in range(len(es)):
        temp_a = temp_a + es[i][2][:-1]
        temp_b = temp_b + p_delta[i][1:]
    k_p_delta = await (ctx.ShareArray(temp_a) * ctx.ShareArray(temp_b))

    a = [0 for _ in range(len(es))]
    b = [0 for _ in range(len(es))]
    c = [0 for _ in range(len(es))]
    t = 0
    for i in range(len(es)):
        c[i] = es[i][2][-1]
        for j in range(len(es[i][2]) - 1):
            c[i] = c[i] + k_p_delta._shares[t + j]
        b[i] = p_delta[i][0]
        a[i] = Field(1) - b[i] - c[i]
        t = t + len(es[i][2]) - 1
    return [a,b,c]

async def batch_PreAnd(ctx, shares, offline):
    result = await batch_PreMul(ctx, shares, offline)
    return result

async def batch_offline_PreMul(ctx, num_of_bits):
    r = []
    s = []
    n = int(((1 + num_of_bits) * num_of_bits)/2)
    print(f' required number of precomputed randoms:{n*2}')
    for _ in range(n):
        r.append(ctx.preproc.get_rand(ctx))
        s.append(ctx.preproc.get_rand(ctx))
    r_array = ctx.ShareArray(r)
    s_array = ctx.ShareArray(s)
    rs = await (r_array * s_array)
    rs_open = await rs.open()

    temp = 0
    result_r = []
    result_s = []
    result_rs_open = []
    for i in range(num_of_bits):
        result_r.append(r[temp: temp + i + 1])
        result_s.append(s[temp: temp + i + 1])
        result_rs_open.append(rs_open[temp: temp + i + 1])
        temp = temp + i + 1

    return result_r, result_s, result_rs_open

async def batch_PreMul(ctx, shares, offline):

    w = []
    z = []
    result = []
    for i in range(1, len(shares) + 1):
        w.append([0 for _ in range(i)])
        z.append([0 for _ in range(i)])
        result.append([0 for _ in range(i)])

    #r, s, u = await batch_offline_PreMul(ctx, len(shares))
    r = offline[0]
    s = offline[1]
    u = offline[2]
    temp_a = []
    temp_b = []
    for i in range(len(shares)):
        temp_a = temp_a + r[i][1:]
        temp_b = temp_b + s[i][: -1]
    v = await (ctx.ShareArray(temp_a) * ctx.ShareArray(temp_b))
    t = 0
    for k in range(len(shares)):
        w[k][0] = r[k][0]
        for i in range(1,len(shares[0])):
            w[k][i] = v._shares(t + i) * (1/u[k][i-1])
        for i in range(len(shares[0])):
            z[k][i] = s[k][i] * (1/u[k][i])
        t = t + len(shares[k])

    temp_a = []
    temp_b = []
    for i in range(len(shares)):
        temp_a = temp_a + w[i]
        temp_b = temp_b + shares[i]  
    m = await (ctx.ShareArray(temp_a) * ctx.ShareArray(temp_b))
    m_open = await m.open()
    t = 0
    for k in range(len(shares)):
        result[k][0] = shares[k][0]
        temp = m_open[k * len(shares[0])]
        for i in range(1, len(shares[0])):
            temp = temp * m_open[t + i]
            result[k][i] = z[k][i] * temp
        t = t + len(shares[k])
    return result

async def Postfix(ctx, shares, offline):
    bit_len = len(shares[0])
    result = [[0 for _ in range(bit_len)], [0 for _ in range(bit_len)], [0 for _ in range(bit_len)]]
    # for i in range(bit_len):
    #     result[0][i], result[1][i], result[2][i] = await carry_prop(ctx, [shares[0][:i+1], shares[1][:i+1], shares[2][:i+1]])
    result = await batch_carry_prop(ctx, [[shares[0][:i+1], shares[1][:i+1], shares[2][:i+1]] for i in range(bit_len)], offline)
    return result

# x is secret share in bit form and y is bit-wise plaintext
async def ComputeCarry(ctx, x, y_binary, offline):
    # y_binary = bin(y.value)[2:]
    # if len(y_binary) < len(x):
    #     y_binary = '0' * (len(x) - len(y_binary)) + y_binary
    # y_binary = y_binary[::-1]

    c = [0 for _ in range(len(x))]
    s = [0 for _ in range(len(x))]
    p = [0 for _ in range(len(x))]
    k = [0 for _ in range(len(x))]

    for i in range(len(x)):
        s[i] = x[i] * Field(int(y_binary[i]))
        p[i] = x[i] + Field(int(y_binary[i])) - Field(2) * x[i] * Field(int(y_binary[i]))
        k[i] = (Field(1) - x[i]) * (Field(1) - Field(int(y_binary[i])))

    f = await Postfix(ctx, [s,p,k], offline)
    return f[0]
# a is bit-wise secret shared and c is plaintext
async def BitSum(ctx, x, y, offline):
    result = [0 for _ in range(len(x) + 1)]

    y_binary = bin(y.value)[2:]
    if len(y_binary) < len(x):
        y_binary = '0' * (len(x) - len(y_binary)) + y_binary
    if len(y_binary) > len(x):
        y_binary = y_binary[len(y_binary) - len(x):]
    y_binary = y_binary[::-1]

    carries = await ComputeCarry(ctx, x, y_binary, offline)
    # add a dummy node at index 0 so that all indexes of carries are right-shifted by 1
    carries.insert(0,"dummy")
    result[0] = x[0] + Field(int(y_binary[0])) - Field(2) * x[0] * Field(int(y_binary[0]))
    for i in range(1, len(x)):
        result[i] = x[i] + Field(int(y_binary[i])) + carries[i] - Field(2) * carries[i+1]
    result[len(x)] = carries[len(x)]
    return result
# share in the imput share and m is the bit-length of share

# Fake offline phase
async def offline_BitDec(ctx, m, k=K):
    r, r_bits = await random2m(ctx, m)
    r_dprime = ctx.Share(random.randint(0, 2 ** (k + KAPPA - m)))
    return r, r_bits, r_dprime

async def BitDec(ctx, share, m, offline, k=K):
    r, r_bits, r_dprime = await offline_BitDec(ctx, m, K)
    r_mask = r + r_dprime * Field(2 ** m)
    c = await (Field(2 ** (k + KAPPA)) + share - r_mask).open()
    result = await BitSum(ctx, r_bits[:m], c, offline)
    return result[:-1]

# result = selection_bits * m_1 + (1 - selection_bits) * m_0
async def OT(ctx, selection_bits, m_0, m_1):
    result = [0 for _ in range(len(selection_bits))]    
    mulpliplicand_1 = selection_bits + selection_bits
    mulpliplicand_2 = m_1 + m_0
    mul_result = await (ctx.ShareArray(mulpliplicand_1) * ctx.ShareArray(mulpliplicand_2))
    for i in range(len(result)):
        result[i] = mul_result._shares[i] + m_0[i] - mul_result._shares[len(selection_bits) + i]
    return result

async def OT_2PC(ctx, message_0, message_1, b):

    m_0 = ctx.Share(message_0)
    m_1 = ctx.Share(message_1)
    b_share = ctx.Share(b)

    m_i = m_0 + (m_1 - m_0) * b_share
    open_m_i = await m_i.open()
    return open_m_i


async def run(ctx, **kwargs):
    k = kwargs["k"]
    V = VSS(ctx)
    shares = [0 for _ in range(k)]
    start = time.time()
    pk = group256.init(G, 1)
    for i in range(k):
        shares[i] = await V.share(i, random.randint(1,10000000))
        pk = pk * group256.deserialize(shares[i].commitments[0])
    stop = time.time()
    print(f"running time for DKG is: {stop - start} seconds")


    # to send a vss share, use "await V.share(dealer, value)"
    # a = await V.share(0, 30)
    # b = await V.share(0, 40)
    # c = a + b
    # open_c = await c.open()
    # print(open_c)

#     a = ctx.Share(2**255 + 2**25 + 2**111 + 2**122 + 2**133 + 2**166 + 2**177 + 2**188 + 2**199 + 2**14 + 2**13 + 2**10 + 2**5 + 2**3 + 2**2)
#     r, s, u = await batch_offline_PreMul(ctx, len(bin(a.v.value)) - 2)
#     start = time.time()
#     b = await BitDec(ctx, a, len(bin(a.v.value)) - 2, [r,s,u])
#     stop = time.time()
#     print(f"total online time for bit Decomposation is: {stop - start} seconds")
#     b_open = await ctx.ShareArray(b).open()
#     print(b_open)

#     selection_bits = [ctx.Share(1) for _ in range(256)]
#     m_0 = [ctx.preproc.get_rand(ctx) for _ in range(256)]
#     m_1 = [ctx.preproc.get_rand(ctx) for _ in range(256)]
#     start = time.time()
#     result = await OT(ctx, selection_bits, m_0, m_1)
#     stop = time.time()
#     print(f"total online time for 256-bit OT is: {stop - start} seconds")



    # x = [ctx.Share(1),ctx.Share(1),ctx.Share(1),ctx.Share(0),ctx.Share(1)]
    # y = Field(6)
    # r = await BitSum(ctx, x, y)
    # r_open = await ctx.ShareArray(r).open()





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
    k = HbmpcConfig.N

    try:
        # pp_elements = FakePreProcessedElements()
        # if HbmpcConfig.my_id == 0:
            
        #     pp_elements.generate_zeros(200, HbmpcConfig.N, HbmpcConfig.t)
        #     pp_elements.generate_triples(150000, HbmpcConfig.N, HbmpcConfig.t)
        #     pp_elements.generate_bits(10000, HbmpcConfig.N, HbmpcConfig.t)
        #     pp_elements.generate_rands(66000, HbmpcConfig.N, HbmpcConfig.t)


        #     # pp_elements.generate_triples(600, HbmpcConfig.N, HbmpcConfig.t)
        #     # pp_elements.generate_rands(600, HbmpcConfig.N, HbmpcConfig.t)
        #     # pp_elements.preprocessing_done()
        # else:
        #     loop.run_until_complete(pp_elements.wait_for_preprocessing())

        loop.run_until_complete(
            _run(HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id, k)
        )
    finally:
        loop.close()
