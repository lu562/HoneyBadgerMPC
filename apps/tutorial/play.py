"""
hbMPC tutorial 2.
Instructions:
   run this with
```
scripts/launch-tmuxlocal.sh apps/tutorial/hbmpc-tutorial-2.py conf/mpc/local
```
"""
import asyncio
import logging
import random
import time
import json
random.seed(562)
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
from  honeybadgermpc.progs.fixedpoint import FixedPoint
mpc_config = {
    MixinConstants.MultiplyShareArray: BeaverMultiplyArrays(),
    MixinConstants.MultiplyShare: BeaverMultiply(),
}

F = 32  # The precision (binary bits)
"""
This implementation of the library is not completely hiding. This leaks information about the bits used in computation which is determinied by the security parameter Kappa.
In particular, we leak O(1/(2^Kappa)) information theorotic bits per operation on a floating point secret.
"""
KAPPA = 16  # Statistical security parameter
K = 64  # Total number of padding bits ()
p = modulus = Subgroup.BLS12_381
BIT_LENGTH = 160
Field = GF(p)
print(p)

def find_divisor(l):
    left = l[0]
    right = l[1]
    divisor = 2 ** (len(l[0]) + len(l[1]))
    if len(l[0]) % 2 == 1:
        divisor = - divisor
    return Field(1) / Field(divisor)

def offline_batch_ltz(ctx, n):
    rs = []
    rs_msb = []
    for i in range(n):
        r_msb = ctx.preproc.get_bit(ctx)
        r_lsbs = Field(random.randint(1,2 ** (K - 1)))
        r = r_msb * (Field(p) - r_lsbs) + (Field(1) - r_msb) * r_lsbs
        rs.append(r)
        rs_msb.append(r_msb)
    return rs,rs_msb


async def batch_ltz(ctx, virables, precomputed_rs, rs_msb):
    num_of_terms = len(virables)
    result = [0 for _ in range(num_of_terms)]
    virables_share = [i.share for i in virables]
    muls =  await (ctx.ShareArray(virables_share) * ctx.ShareArray(precomputed_rs))

    xr_open = await muls.open()
    for i in range(num_of_terms):
        sign = Field(0)
        if xr_open[i].value >= p/2 :
            sign = Field(1)
        result[i] = rs_msb[i] + sign - 2 * rs_msb[i] * sign
    return result

def decision_tree_offline(ctx, num_of_terms, num_of_virables):
    result = [[] for _ in range(num_of_terms)]
    product = [ctx.field(1) for _ in range(num_of_terms)]

    for i in range(num_of_terms):
        p = ctx.field(1)
        for j in range(num_of_virables):
            a = ctx.field(random.randint(1,50))    
            result[i].append(ctx.Share(a)) 
            product[i] = product[i] * (1/a)

        product[i] = ctx.Share(product[i])
    return result,product


async def batch_decision_tree_eval(ctx, terms, precompute_randoms, product):
    num_of_terms = len(terms)
    num_of_virables = len(terms[0])
    combined_terms =  [x for j in terms for x in j]
    combined_precompute_randoms =  [x for j in precompute_randoms for x in j]
    result = [0 for _ in range(num_of_terms)]
      
    start =  time.time()
    m = await (ctx.ShareArray(combined_terms) * ctx.ShareArray(combined_precompute_randoms))
    t1 = time.time()
    open_m = await m.open()
    t2 = time.time()
    for i in range(num_of_terms):
        mul = ctx.field(1)
        for j in range(num_of_virables):
            mul = mul * open_m[i * num_of_virables + j]
        result[i] = mul * product[i]
    t3 = time.time()
    print(f'time for mul: {t1 - start}')
    print(f'time for mul: {t2 -t1}')
    print(f'time for mul: {t3 - t2}')
    return result

def random2m(ctx, m):
    result = ctx.Share(0)
    bits = []
    for i in range(m):
        bits.append(ctx.preproc.get_bit(ctx))
        result = result + Field(2) ** i * bits[-1]

    return result, bits

async def wrap_around(ctx, x_bits, r_bits):
    w = [0 for _ in range(BIT_LENGTH)]
    sum_w = [0 for _ in range(BIT_LENGTH)]
    c = [0 for _ in range(BIT_LENGTH)]
    for i in range(BIT_LENGTH):
        w[i] = x_bits[i] + ctx.field(int(r_bits[i])) - ctx.field(2) * x_bits[i] * ctx.field(int(r_bits[i]))
    temp_sum = ctx.field(0)
    sum_w[BIT_LENGTH - 1] = ctx.field(0)
    for i in range(BIT_LENGTH - 1):
        temp_sum = temp_sum + w[BIT_LENGTH - 1 - i]
        sum_w[BIT_LENGTH - 2 - i] = temp_sum
    for i in range(BIT_LENGTH):
        c[i] = ctx.field(int(r_bits[i])) - x_bits[i] + ctx.field(1) + sum_w[i]
    c_open = await ctx.ShareArray(c).open()
    print(c_open)
    # more steps needed to hide c.
    result = 0
    for i in range(BIT_LENGTH):
        if c_open[i].value == 0:
            return 1
    return 0

async def ltz2(ctx, a, precomputed_x, x_bits):
    y = ctx.field(2) * a # result is LSB(y)
    r = await (y + precomputed_x).open()
    x_open = await  precomputed_x.open()
    print(bin(x_open.value).replace('0b', '')[::-1])
    r_bits = bin(r.value).replace('0b', '')[::-1]

    if len(r_bits) < BIT_LENGTH:
        r_bits = r_bits + '0' * (BIT_LENGTH - len(r_bits))
    # wrap around
    wrap = await wrap_around(ctx, x_bits, r_bits)
    print(wrap)

    xr =  x_bits[0] + ctx.field(int(r_bits[0])) - ctx.field(2) * x_bits[0] * ctx.field(int(r_bits[0]))
    if wrap == 0:
        return xr
    else:
        return ctx.field(1) - xr

async def batch_wrap_around(ctx, x_bits, r_bits):
    w = [[0 for _ in range(BIT_LENGTH)] for _ in range(len(x_bits))]
    sum_w = [[0 for _ in range(BIT_LENGTH)] for _ in range(len(x_bits))]
    c = [[0 for _ in range(BIT_LENGTH)] for _ in range(len(x_bits))]
    for n in range(len(x_bits)):

        for i in range(BIT_LENGTH):
            w[n][i] = x_bits[n][i] + ctx.field(int(r_bits[n][i])) - ctx.field(2) * x_bits[n][i] * ctx.field(int(r_bits[n][i]))
        temp_sum = ctx.field(0)
        sum_w[n][BIT_LENGTH - 1] = ctx.field(0)
        for i in range(BIT_LENGTH - 1):
            temp_sum = temp_sum + w[n][BIT_LENGTH - 1 - i]
            sum_w[n][BIT_LENGTH - 2 - i] = temp_sum
        for i in range(BIT_LENGTH):
            c[n][i] = ctx.field(int(r_bits[n][i])) - x_bits[n][i] + ctx.field(1) + sum_w[n][i]

    open_array = [i for item in c for i in item]

    c_open = await ctx.ShareArray(open_array).open()
    result = []
    # more steps needed to hide c.
    for i in range(len(x_bits)):
        temp = 0
        for j in range(BIT_LENGTH):
            if c_open[i * BIT_LENGTH + j].value == 0:
                temp = 1
                break
        result.append(temp)

    return result

async def batch_ltz2(ctx, a, precomputed_x, x_bits):
    y = [ctx.field(2) * v for v in a ]# result is LSB(y)
    r = await ctx.ShareArray([y[i] + precomputed_x[i] for i in range(len(a))]).open()

    r_bits = [bin(r[i].value).replace('0b', '')[::-1] for i in range(len(a))]
    for i in range(len(a)):
        if len(r_bits[i]) < BIT_LENGTH:
            r_bits[i] = r_bits[i] + '0' * (BIT_LENGTH - len(r_bits[i]))
    # wrap around
    wrap = await batch_wrap_around(ctx, x_bits, r_bits)

    xr =  [(x_bits[i][0] + ctx.field(int(r_bits[i][0])) - ctx.field(2) * x_bits[i][0] * ctx.field(int(r_bits[i][0]))) for i in range(len(a))]
    result = []
    for i in range(len(a)):
        if wrap[i] == 0:
            result.append(xr[i])
        else:
            result.append(ctx.field(1) - xr[i])
    return result


async def run(ctx, **kwargs):

    # # read json from files
    # poly = {}
    # comparison = {}
    # values = {}
    # divisors = {}
    # poly_j = ''
    # comparison_j = ''
    # values_j = ""
    # with open('/usr/src/HoneyBadgerMPC/apps/tutorial/json_poly.json', 'r') as json_file:
    #     poly_j =json_file.readline()
    # with open('/usr/src/HoneyBadgerMPC/apps/tutorial/json_comparison.json', 'r') as json_file:
    #     comparison_j = json_file.readline()
    # with open('/usr/src/HoneyBadgerMPC/apps/tutorial/json_value.json', 'r') as json_file:
    #     values_j = json_file.readline()

    # poly = json.loads(poly_j)
    # comparison = json.loads(comparison_j)
    # values = json.loads(values_j)
    # for key,value in poly.items():
    #     divisors[key] = find_divisor(value)
    # rs,rs_msb = offline_batch_ltz(ctx, len(comparison))
    # # get the number of virables in each poly
    # a = random.sample(poly.keys(), 1)  
    # b = a[0] 
    # precompute_randoms, product = decision_tree_offline(ctx, len(poly), len(poly[b][0]) + len(poly[b][1]) + 1)

    # # online phase
    # test_input = [FixedPoint(ctx,1), FixedPoint(ctx,2), FixedPoint(ctx,3), FixedPoint(ctx,5), FixedPoint(ctx,4), FixedPoint(ctx,6), FixedPoint(ctx,1), FixedPoint(ctx,0),]
    # print("test input from client: (1, 2, 3, 5, 4, 6, 1, 0)")    
    # virables = []
    # ids = []
    # start =  time.time()
    # for node_id, terms in comparison.items():
    #     ids.append(int(node_id))
    #     virables.append(FixedPoint(ctx,terms[1]) - test_input[terms[0]])
    # logging.info("start secure comparison")
    # comparison_result = await batch_ltz(ctx, virables, rs, rs_msb)
    # logging.info("secure comparison finished")
    # for i in range(len(comparison_result)):
    #     comparison_result[i] = (comparison_result[i] - Field(1)/Field(2)) * Field(2)

    # minus_one_terms = [ (i - Field(1)) for i in comparison_result]
    # plus_one_terms = [ (i + Field(1)) for i in comparison_result]

    # poly_terms = []
    # poly_id = []
    # # poly_print = []
    # # for key,value in poly.items():
    # #     s = ''
    # #     for i in value[0]:
    # #         s = s + f"(x{i} - 1)"
    # #     for i in value[1]:
    # #         s = s + f"(x{i} + 1)"
    # #     # s = s + f"/{divisors[key]}"
    # #     poly_print.append(s)    

    # # print("To evaluate the decision tree, we need to evaluate 256 polynomial terms, below are first ten of them:")
    # # for i in range(10):
    # #     print(poly_print[i])



    # for key,value in poly.items():
    #     poly_id.append(key)
    #     terms = []
    #     for i in value[0]:
    #         terms.append(minus_one_terms[ids.index(int(i))])
    #     for i in value[1]:
    #         terms.append(plus_one_terms[ids.index(int(i))])
    #     terms.append(values[int(key)])
    #     poly_terms.append(terms)

    # logging.info("start evaluation")
    # poly_results = await batch_decision_tree_eval(ctx, poly_terms, precompute_randoms, product)
    # logging.info("evaluation finished")
    # middle = time.time()
    # for i in range(len(poly_results)):
    #     poly_results[i] = poly_results[i] * divisors[poly_id[i]]

    # stop =  time.time()
    # # logging.info(f"time for division: {stop - middle}")
    # logging.info(f"total online time: {stop - start}")

    # open_result = await ctx.ShareArray(poly_results).open() 
    # # print(open_result)
    # for i in range(len(open_result)):
    #     if open_result[i].value == 1:
    #         print(f"The evaluation result is stored in leaf node {i}.")


    # logging.info("Starting _prog")
    # a = FixedPoint(ctx, 99999999.5)
    # b = FixedPoint(ctx, -3.8)
    # # A = await a.open()  # noqa: F841, N806
    # # B = await b.open()  # noqa: F841, N806
    # # AplusB = await (a + b).open()  # noqa: N806
    # # AminusB = await (a - b).open()  # noqa: N806
    # # AtimesB = await (await a.__mul__(b)).open()  # noqa: N806
    # # logging.info("Starting less than")
    # # # AltB = await (await a.lt(b)).open()  # noqa: N806
    # # # BltA = await (await b.lt(a)).open()  # noqa: N806
    # s_time = time.time()
    # for _ in range(1):
    #     AltB = await a.ltz()
    # e_time = time.time()
    # logging.info(f"total online time: {e_time - s_time}")
    # # BltA = await (await b.new_ltz()).open() 
    # # logging.info("done")
    # # logging.info(f"A:{A} B:{B} A-B:{AminusB} A+B:{AplusB}")
    # # logging.info(f"A*B:{AtimesB} A<B:{AltB} B<A:{BltA}")
    # logging.info("Finished _prog")

    k = 500
    # Our method
    rs = [ctx.Share(10)] * k
    rs_msb = [ctx.Share(0)] * k
    variables = [FixedPoint(ctx,5123123)] * k
    s_time = time.time()
    comparison_result = await batch_ltz(ctx, variables, rs, rs_msb)
    e_time = time.time()
    logging.info(f"total online time for our method: {e_time - s_time}")
    # #SecureNN method
    # r, r_bits = random2m(ctx, BIT_LENGTH)
    # # r = ctx.Share(10)
    # # r_bits = [ctx.Share(0), ctx.Share(1), ctx.Share(0), ctx.Share(1)] + [ctx.Share(0)] * 156
    # k = 100
    # s_time = time.time()
    # # result = await ltz2(ctx, ctx.Share(5123123), r, r_bits)
    # result = await batch_ltz2(ctx, [ctx.Share(5123123)] * k, [r] * k, [r_bits] * k)
    # e_time = time.time()

    # result_open = await ctx.ShareArray(result).open()
    # logging.info(f"total online time for secureNN method: {e_time - s_time}")



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
        # pp_elements = FakePreProcessedElements()
        # if HbmpcConfig.my_id == 0:
            
        #     pp_elements.generate_zeros(200, HbmpcConfig.N, HbmpcConfig.t)
        #     pp_elements.generate_triples(200, HbmpcConfig.N, HbmpcConfig.t)
        #     pp_elements.generate_bits(200, HbmpcConfig.N, HbmpcConfig.t)
        #     pp_elements.preprocessing_done()

            
        #     # pp_elements.generate_zeros(20000, HbmpcConfig.N, HbmpcConfig.t)
        #     # pp_elements.generate_triples(260000, HbmpcConfig.N, HbmpcConfig.t)
        #     # pp_elements.generate_bits(20000, HbmpcConfig.N, HbmpcConfig.t)
        #     # pp_elements.preprocessing_done()

        # else:
        #     loop.run_until_complete(pp_elements.wait_for_preprocessing())

        loop.run_until_complete(
            _run(HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id, k)
        )
    finally:
        loop.close()
