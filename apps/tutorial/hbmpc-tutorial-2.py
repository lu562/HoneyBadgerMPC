import asyncio
import logging
import random
import threading
import time
import math
from honeybadgermpc.preprocessing import (
    PreProcessedElements as FakePreProcessedElements,
)
from honeybadgermpc.field import GFElement
from honeybadgermpc.progs.mixins.dataflow import Share
from honeybadgermpc.utils.task_pool import TaskPool
from honeybadgermpc.progs.mixins.share_arithmetic import (
    BeaverMultiply,
    BeaverMultiplyArrays,
    MixinConstants,
)
mpc_config = {
    MixinConstants.MultiplyShareArray: BeaverMultiplyArrays(),
    MixinConstants.MultiplyShare: BeaverMultiply(),
}
random.seed(5622)
# non multi threading if MODE = 0
MODE = 1 
total_add_time = 0
total_mul_time = 0
total_communicate_time = 0

def generate_beaver_matrix_hack(ctx, k, m, n):
    A_hack = [[i + j for j in range(m)] for i in range(k)]
    B_hack = [[i + 2 * j for j in range(n)] for i in range(m)]
    C_hack = [[0 for _ in range(n)]for _ in range(k)]
    for i in range(k):
        for j in range(n):
            for t in range(m): 
                C_hack[i][j] = C_hack[i][j] +  A_hack[i][t] * B_hack[t][j]
    A = [[ctx.Share(A_hack[i][j]) for j in range(m)] for i in range(k)]
    B = [[ctx.Share(B_hack[i][j]) for j in range(n)] for i in range(m)]
    C = [[ctx.Share(C_hack[i][j]) for j in range(n)] for i in range(k)]

    return A, B, C

def generate_beaver_triple_matrix_hack(ctx, k, m, n, q):
    A_hack = [[i + j for j in range(m)] for i in range(k)]
    B_hack = [[i + 2 * j for j in range(n)] for i in range(m)]
    C_hack = [[i + 3 * j for j in range(q)]for i in range(n)]
    D_hack = [[0 for _ in range(q)]for _ in range(k)]
    AB_hack = [[0 for _ in range(n)]for _ in range(k)]
    for i in range(k):
        for j in range(n):
            for t in range(m): 
                AB_hack[i][j] = AB_hack[i][j] +  A_hack[i][t] * B_hack[t][j]
    for i in range(k):
        for j in range(q):
            for t in range(n): 
                D_hack[i][j] = D_hack[i][j] +  AB_hack[i][t] * C_hack[t][j]
    A = [[ctx.Share(A_hack[i][j]) for j in range(m)] for i in range(k)]
    B = [[ctx.Share(B_hack[i][j]) for j in range(n)] for i in range(m)]
    C = [[ctx.Share(C_hack[i][j]) for j in range(q)] for i in range(n)]
    D = [[ctx.Share(D_hack[i][j]) for j in range(q)] for i in range(k)]

    return A, B, C, D

# these function is used to open a batch of matrices with the same dimension
async def batch_matrix_open(ctx, M):
    global total_communicate_time
    k = len(M[0])
    c = len(M[0][0])
    array = [0 for _ in range(k * c * len(M))]
    open_array = [0 for _ in range(k * c * len(M))]
    for m in range(len(M)):
        for i in range(k):
            for j in range(c):
                array[m * k * c + i * k + j] = M[m][i][j]
    res = [[[0 for _ in range(c)] for _ in range(k)] for _ in range(len(M))]  
    start = time.time() 
    await asyncio.sleep(0.1)
    open_array  = await ctx.ShareArray(array).open()
    stop = time.time() 
    total_communicate_time = total_communicate_time + stop - start
    for m in range(len(M)):
        for i in range(k):
            for j in range(c):
                res[m][i][j] = open_array[m * k * c + i * k + j]
    return res

def matrix_addition(matrix_a, matrix_b):
    m = len(matrix_a)
    n = len(matrix_a[0])
    output = [[0 for _ in range(n)]for _ in range(m)]
    for i in range(m):
        for j in range(n):
            output[i][j] = matrix_a[i][j] + matrix_b[i][j]
    return output

def matrix_sub(matrix_a, matrix_b):
    m = len(matrix_a)
    n = len(matrix_a[0])
    output = [[ctx.Share(0) for _ in range(n)]for _ in range(m)]
    for i in range(m):
        for j in range(n):
            output[i][j] = matrix_a[i][j] - matrix_b[i][j]
    return output

def matrix_mul(ctx, matrix_a, matrix_b):
    k = len(matrix_a)
    m = len(matrix_a[0])
    n = len(matrix_b[0])    
    # plain matrix is k*m and share matrix is m*n, output matrix is k*n
    output = [[ctx.field(0) for _ in range(n)]for _ in range(k)]
    for i in range(k):
        for j in range(n):
            for t in range(m):
                output[i][j] = output[i][j] + matrix_a[i][t] * matrix_b[t][j]
    return output

# this method support non-square matrices.
async def beaver_mul_matrix(ctx, X, Y, A, B, C):
    k = len(X)
    m = len(X[0])
    n = len(Y[0])
    D = await batch_cpp_matrix_sub(ctx, [X], [A])
    E = await batch_cpp_matrix_sub(ctx, [Y], [B])
    o = await batch_matrix_open(ctx, [D[0],E[0]])
    D_open = o[0]
    E_open = o[1]
    res = [[[ctx.Share(0) for _ in range(n)]for _ in range(k)]]

    DE = await batch_cpp_matrix_mul(ctx, [D_open], [E_open])
    res = await batch_cpp_matrix_add(ctx, res, DE)
    AE = await batch_cpp_matrix_mul(ctx, [A], [E_open])
    res = await batch_cpp_matrix_add(ctx, res, AE)
    DB = await batch_cpp_matrix_mul(ctx, [D_open], [B])
    res = await batch_cpp_matrix_add(ctx, res, DB)  
    res = await batch_cpp_matrix_add(ctx, res, [C])

    return res[0]

async def beaver_mul_three_matrix(ctx, X, Y, Z):
    k = len(X)
    m = len(X[0])
    n = len(Y[0])
    q = len(Z[0])
    A,B,C,D = generate_beaver_triple_matrix_hack(ctx, k, m, n, q)    

    X_minus_A = [[X[i][j] - A[i][j] for j in range(m)] for i in range(k)]
    Y_minus_B = [[Y[i][j] - B[i][j] for j in range(n)] for i in range(m)]
    Z_minus_C = [[Z[i][j] - C[i][j] for j in range(q)] for i in range(n)]

    batch = [X_minus_A, Y_minus_B, Z_minus_C]
    o = await batch_matrix_open(ctx, batch)
    X_minus_A_open = o[0]
    Y_minus_B_open = o[1]
    Z_minus_C_open = o[2]

    res = [[0 for _ in range(q)]for _ in range(k)]

    # E = (X-A)(Y-B) F = (X-A)(Y-B)(Z-C)
    E = matrix_mul(ctx,  X_minus_A_open, Y_minus_B_open)
    F = matrix_mul(ctx,  E, Z_minus_C_open)
    res = matrix_addition(res, F)

    A1,B1,C1 = generate_beaver_matrix_hack(ctx, m, n, q)
    BZ = await beaver_mul_matrix(ctx, B, Z, A1,B1,C1)
    X_minus_A_BZ = matrix_mul_plain_share(ctx, X_minus_A_open , BZ)
    res = matrix_addition(res, X_minus_A_BZ)

    A1,B1,C1 = generate_beaver_matrix_hack(ctx,k, m, n)
    AY = await beaver_mul_matrix(ctx, A, Y, A1,B1,C1 )
    AY_Z_minus_C = matrix_mul_plain_share(ctx, AY , Z_minus_C_open)
    res = matrix_addition(res, AY_Z_minus_C)
    # calculate X(Y - B)C
    A1,B1,C1 = generate_beaver_matrix_hack(ctx, k, n, q)
    X_Y_minus_B = matrix_mul_plain_share(ctx, X , Y_minus_B_open)
    X_Y_minus_B_C = await beaver_mul_matrix(ctx, X_Y_minus_B, C, A1, B1, C1)
    res = matrix_addition(res, X_Y_minus_B_C)

    res = matrix_addition(res, D)
    return res

def matrix_mul_plain_share(ctx, matrix_a, matrix_b):
    # plain matrix is k*m and share matrix is m*n, output matrix is k*n
    k = len(matrix_a)
    m = len(matrix_a[0])
    n= len(matrix_b[0])
    output = [[ctx.Share(0) for _ in range(k)]for _ in range(n)]
    for i in range(k):
        for j in range(n):
            for t in range(m):          

                output[i][j] = output[i][j] + matrix_a[i][t] * matrix_b[t][j]

    return output

# this only works when all input matrix have the same dimension
# also to speed up offline phase, hack is used here, I use the same triples to multiply all X&Y.
async def batch_beaver_mul_matrix(ctx, X, Y, A, B, C):

    num_of_matrix = len(X)
    k = len(X[0])
    m = len(X[0][0])
    n = len(Y[0][0])

    D = await batch_cpp_matrix_sub(ctx, X, [A for _ in range(num_of_matrix)])
    E = await batch_cpp_matrix_sub(ctx, Y, [B for _ in range(num_of_matrix)])

    start_open =  time.time()
    o = await batch_matrix_open(ctx, D+E)
    stop_open =  time.time()
    logging.info(f"Inside batch beaver matrix mul: time for opening (X - A)(Y - B)): {stop_open - start_open}")

    D_open = [0 for _ in range(num_of_matrix)]
    E_open = [0 for _ in range(num_of_matrix)]
    DE = [0 for _ in range(num_of_matrix)]
    AE = [0 for _ in range(num_of_matrix)]
    DB = [0 for _ in range(num_of_matrix)]
    for i in range(num_of_matrix):
        D_open[i] = o[i]
    for i in range(num_of_matrix):
        E_open[i] = o[i + num_of_matrix]

    res = [[[ctx.Share(0) for _ in range(n)]for _ in range(k)]for _ in range(num_of_matrix)]
    DE = await batch_cpp_matrix_mul(ctx, D_open, E_open)
    res = await batch_cpp_matrix_add(ctx, res, DE)
    AE = await batch_cpp_matrix_mul(ctx, [A for _ in range(num_of_matrix)], E_open)
    res = await batch_cpp_matrix_add(ctx, res, AE) 
    DB = await batch_cpp_matrix_mul(ctx, D_open, [B for _ in range(num_of_matrix)])
    res = await batch_cpp_matrix_add(ctx, res, DB)
    res = await batch_cpp_matrix_add(ctx, res, [C for _ in range(num_of_matrix)])
    return res

async def batch_beaver_mul_three_matrix_with_precomputation(ctx, X, Y, Z, super_triple, normal_triple):
    num_of_matrix = len(X)
    A = normal_triple[0]
    B = normal_triple[1]
    C = normal_triple[2]

    start =  time.time() 
    res = await batch_beaver_mul_matrix(ctx, X, Y, A, B, C)
    res = await batch_beaver_mul_matrix(ctx, res, Z, A, B, C)
    stop =  time.time() 
    logging.info(f"actual time to run 2 beaver matrix mul: {stop - start}")
    return res

# batch method only support square matrices.
async def backup_batch_beaver_mul_three_matrix_with_precomputation(ctx, X, Y, Z, super_triple, normal_triple):
    num_of_matrix = len(X)
    k = len(X[0])
    m = len(X[0][0])
    n = len(Y[0][0])
    q = len(Z[0][0])
    # tricks used here, to speed up the testing, the same supertriple and beaver triple is used for all matrices multiplication
    A = super_triple[0]
    B = super_triple[1]
    C = super_triple[2]
    D = super_triple[3]
    X_minus_A = [0 for _ in range(num_of_matrix)]
    Y_minus_B = [0 for _ in range(num_of_matrix)]
    Z_minus_C = [0 for _ in range(num_of_matrix)]
    for nn in range(num_of_matrix):
        X_minus_A[nn] = [[X[nn][i][j] - A[i][j] for j in range(m)] for i in range(k)]
        Y_minus_B[nn] = [[Y[nn][i][j] - B[i][j] for j in range(n)] for i in range(m)]
        Z_minus_C[nn] = [[Z[nn][i][j] - C[i][j] for j in range(q)] for i in range(n)]
    batch = X_minus_A + Y_minus_B + Z_minus_C
    startt2 = time.time()
    o = await batch_matrix_open(ctx, batch)
    stopt2 = time.time()
    logging.info(f"time for opening all (X-A),(Y-B),(Z-C): {stopt2 - startt2}")
    X_minus_A_open = o[0: num_of_matrix]
    Y_minus_B_open = o[num_of_matrix : num_of_matrix * 2]
    Z_minus_C_open = o[num_of_matrix * 2 :]

    res = [[[ctx.Share(0) for _ in range(q)] for _ in range(k)] for _ in range(num_of_matrix)]
    # E = (X-A)(Y-B) F = (X-A)(Y-B)(Z-C)
    E = [0 for _ in range(num_of_matrix)]
    F = [0 for _ in range(num_of_matrix)]
    X_Y_minus_B = [0 for _ in range(num_of_matrix)]
    BZ = [0 for _ in range(num_of_matrix)]
    AY = [0 for _ in range(num_of_matrix)]
    X_Y_minus_B_C = [0 for _ in range(num_of_matrix)]
    X_minus_A_BZ = [0 for _ in range(num_of_matrix)]
    AY_Z_minus_C = [0 for _ in range(num_of_matrix)]
    startt3 = time.time()
    E = await batch_cpp_matrix_mul(ctx, X_minus_A_open, Y_minus_B_open)
    F = await batch_cpp_matrix_mul(ctx, E, Z_minus_C_open)
    X_Y_minus_B = await batch_cpp_matrix_mul(ctx, X , Y_minus_B_open)

    stopt3 = time.time()
    logging.info(f"time for computing all share matrices and opened matrices(3 mul): { stopt3 - startt3 }")
    startt33 = time.time()
    res = await batch_cpp_matrix_add(ctx, res, F)
    stopt33 = time.time()
    logging.info(f"time for computing all share matrices and opened matrices(1 add): { stopt33 - startt33 }")

    # To save testing time I use hacked trick here, I use the same beaver triples to do 3 multiplication
    A2 = normal_triple[0]
    B2 = normal_triple[1]
    C2 = normal_triple[2]

    # batch method starting from here
    batch_X = []
    batch_Y = []
    for nn in range(num_of_matrix):
        batch_X = batch_X + [B,A,X_Y_minus_B[nn]]
        batch_Y = batch_Y + [Z[nn],Y[nn],C]
    startt4 = time.time()
    o = await batch_beaver_mul_matrix(ctx, batch_X, batch_Y, A2, B2, C2)
    stopt4 = time.time()
    logging.info(f"time for batch beaver matices mul: {stopt4 - startt4}")
    startt5 = time.time()  
    for nn in range(num_of_matrix):
        BZ[nn] = o[0 + 3 * nn]
        AY[nn] = o[1 + 3 * nn]
        X_Y_minus_B_C[nn] = o[2 + 3 * nn]

    X_minus_A_BZ = await batch_cpp_matrix_mul(ctx, X_minus_A_open , BZ)
    res = await batch_cpp_matrix_add(ctx, res, X_minus_A_BZ)
    AY_Z_minus_C = await batch_cpp_matrix_mul(ctx, AY , Z_minus_C_open)

    res = await batch_cpp_matrix_add(ctx, res, AY_Z_minus_C)
    res = await batch_cpp_matrix_add(ctx, res, X_Y_minus_B_C)
    res = await batch_cpp_matrix_add(ctx, res,  [D for _ in range(num_of_matrix)])
    stopt5 = time.time()
    logging.info(f"time for computing all share matrices and opened matrices(part 2): {stopt5 - startt5}")
    return res

# codes for calculating matrix inverse
def determinant(A, total=0):
    indices = list(range(len(A)))
    
    if len(A) == 2 and len(A[0]) == 2:
        val = A[0][0] * A[1][1] - A[1][0] * A[0][1]
        return val

    for fc in indices:
        As = copy_matrix(A)
        As = As[1:]
        height = len(As)
        builder = 0

        for i in range(height):
            As[i] = As[i][0:fc] + As[i][fc+1:]

        sign = (-1) ** (fc % 2)
        sub_det = determinant(As)
        total += A[0][fc] * sign * sub_det

    return total
        
def zeros_matrix(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have
        :returns: list of lists that form the matrix.
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0)

    return M

def identity_matrix(n):
    """
    Creates and returns an identity matrix.
        :param n: the square size of the matrix
        :returns: a square identity matrix
    """
    I = zeros_matrix(n, n)
    for i in range(n):
        I[i][i] = 1

    return I

def copy_matrix(M):
    """
    Creates and returns a copy of a matrix.
        :param M: The matrix to be copied
        :return: The copy of the given matrix
    """
    rows = len(M)
    cols = len(M[0])

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(rows):
            MC[i][j] = M[i][j]

    return MC

def print_matrix(M):
    """
    docstring here
        :param M: The matrix to be printed
    """
    for row in M:
        print([round(x,3)+0 for x in row])

def transpose(M):
    """
    Creates and returns a transpose of a matrix.
        :param M: The matrix to be transposed
        :return: the transpose of the given matrix
    """
    rows = len(M)
    cols = len(M[0])

    MT = zeros_matrix(cols, rows)

    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT

def invert_matrix(A, tol=None):
    """
    Returns the inverse of the passed in matrix.
        :param A: The matrix to be inversed
        :return: The inverse of the matrix A
    """
    n = len(A)
    AM = copy_matrix(A)
    I = identity_matrix(n)
    IM = copy_matrix(I)
    # Section 3: Perform row operations
    indices = list(range(n)) # to allow flexible row referencing ***
    for fd in range(n): # fd stands for focus diagonal
        fdScaler = 1 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse. 
        for j in range(n): # Use j to indicate column looping.
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
        # SECOND: operate on all rows except fd row as follows:
        for i in indices[0:fd] + indices[fd+1:]: # *** skip row with fd in it.
            crScaler = AM[i][fd] # cr stands for "current row".
            for j in range(n): # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]
    return IM

# Codes about multiplying arbitrary number of matrices

def matrix_inverse_hack(ctx, k):
    
    A_hack = [[ctx.field(random.randint(1,50)) for j in range(k)] for i in range(k)]
    A_inverse_hack = invert_matrix(A_hack)

    A = [[ctx.Share(A_hack[i][j]) for j in range(k)] for i in range(k)]
    A_inverse = [[ctx.Share(A_inverse_hack[i][j]) for j in range(k)] for i in range(k)]
    return A, A_inverse

def offline_multi_matrix_multiply(ctx, k, n):
    R = []
    R_inverse = []
    for i in range(n + 1):
        a, a_inverse = matrix_inverse_hack(ctx, k)
        R.append(a)
        R_inverse.append(a_inverse)
    return R, R_inverse

async def run_command_sync(command):
    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    logging.debug(f"Command:{command}")
    logging.debug(f"Output: {stdout}")
    if len(stderr):
        logging.info(f"Error: {stderr}")

async def batch_cpp_matrix_mul(ctx, A, B):
    global total_mul_time
    num_of_batch = len(A)
    # honeybadgermpc.field.GFElement
    A_type = (type(A[0][0][0]) == GFElement)
    B_type = (type(B[0][0][0]) == GFElement)

    result = [[[0 for _ in range(len(A[0][0]))] for _ in range(len(A[0]))] for _ in range(num_of_batch)]
    # write matrices A  into files
    file_name = f"matrix_{ctx.myid}_A.input"
    file_path = f"sharedata/{file_name}"
    with open(file_path, "w") as f:
        for i in range(num_of_batch):        
            print(ctx.field.modulus, file=f)
            print(len(A[i]), file=f)
            print(len(A[i][0]), file=f)
            if(A_type):
                for ii in range(len(A[i])):
                    for jj in range(len(A[i][0])):
                        print(A[i][ii][jj].value, file=f)
            else:
                 for ii in range(len(A[i])):
                    for jj in range(len(A[i][0])):
                        print(A[i][ii][jj].v.value, file=f)
    # write matrices B into files
    file_name = f"matrix_{ctx.myid}_B.input"
    file_path = f"sharedata/{file_name}"
    with open(file_path, "w") as f:
        for i in range(num_of_batch):
            print(ctx.field.modulus, file=f)
            print(len(B[i]), file=f)
            print(len(B[i][0]), file=f)
            if(B_type):
                for ii in range(len(B[i])):
                    for jj in range(len(B[i][0])):
                        print(B[i][ii][jj].value, file=f)
            else:
                 for ii in range(len(B[i])):
                    for jj in range(len(B[i][0])):
                        print(B[i][ii][jj].v.value, file=f)  
    # do computation
    runcmd = f"./apps/tutorial/cpp/multi_matrix_mul {num_of_batch} {ctx.myid} {MODE}"
    await run_command_sync(runcmd)

    # read C++ benchmark
    file_name = f"benchmark_mul.log"
    file_path = f"sharedata/{file_name}"
    cpp_time = 0;
    with open(file_path, "r") as f:
        cpp_time = f.readline()
    # logging.info(f"Batch_cpp_mul, pure computation time each call: {cpp_time}")
    # total_mul_time = total_mul_time + float(cpp_time)
    # logging.info(f"mul time now:: {total_mul_time}")
    #load result from files
    file_name = f"matrix_{ctx.myid}_C.output"
    file_path = f"sharedata/{file_name}"
    with open(file_path, "r") as f:
        for i in range(num_of_batch):
            assert ctx.field.modulus == int(f.readline())
            row = int(f.readline())
            column = int(f.readline())
            if A_type and B_type:
                for r in range(row):
                    for c in range(column):
                        result[i][r][c] = ctx.field(int(f.readline()))
            else:
                for r in range(row):
                    for c in range(column):
                        result[i][r][c] = ctx.Share(int(f.readline()))                
    return result

async def batch_cpp_matrix_add(ctx, A, B):
    global total_add_time
    num_of_batch = len(A)
    # honeybadgermpc.field.GFElement
    A_type = (type(A[0][0][0]) == GFElement)
    B_type = (type(B[0][0][0]) == GFElement)

    result = [[[0 for _ in range(len(A[0][0]))] for _ in range(len(A[0]))] for _ in range(num_of_batch)]
    # write matrices A  into files
    file_name = f"matrix_{ctx.myid}_A.input"
    file_path = f"sharedata/{file_name}"
    with open(file_path, "w") as f:
        for i in range(num_of_batch):        
            print(ctx.field.modulus, file=f)
            print(len(A[i]), file=f)
            print(len(A[i][0]), file=f)
            if(A_type):
                for ii in range(len(A[i])):
                    for jj in range(len(A[i][0])):
                        print(A[i][ii][jj].value, file=f)
            else:
                 for ii in range(len(A[i])):
                    for jj in range(len(A[i][0])):
                        print(A[i][ii][jj].v.value, file=f)
    # write matrices B into files
    file_name = f"matrix_{ctx.myid}_B.input"
    file_path = f"sharedata/{file_name}"
    with open(file_path, "w") as f:
        for i in range(num_of_batch):
            print(ctx.field.modulus, file=f)
            print(len(B[i]), file=f)
            print(len(B[i][0]), file=f)
            if(B_type):
                for ii in range(len(B[i])):
                    for jj in range(len(B[i][0])):
                        print(B[i][ii][jj].value, file=f)
            else:
                 for ii in range(len(B[i])):
                    for jj in range(len(B[i][0])):
                        print(B[i][ii][jj].v.value, file=f)  
    # do computation
    runcmd = f"./apps/tutorial/cpp/multi_matrix_add {num_of_batch} {ctx.myid} {MODE}"
    await run_command_sync(runcmd)

    # read C++ benchmark
    file_name = f"benchmark_add.log"
    file_path = f"sharedata/{file_name}"
    cpp_time = 0;
    with open(file_path, "r") as f:
        cpp_time = f.readline()

    # total_add_time = total_add_time + float(cpp_time)
    # logging.info(f"add time now:: {total_add_time}")

    #load result from files
    file_name = f"matrix_{ctx.myid}_C.output"
    file_path = f"sharedata/{file_name}"
    with open(file_path, "r") as f:
        for i in range(num_of_batch):
            assert ctx.field.modulus == int(f.readline())
            row = int(f.readline())
            column = int(f.readline())
            if A_type and B_type:
                for r in range(row):
                    for c in range(column):
                        result[i][r][c] = ctx.field(int(f.readline()))
            else:
                for r in range(row):
                    for c in range(column):
                        result[i][r][c] = ctx.Share(int(f.readline()))                
    return result

async def batch_cpp_matrix_sub(ctx, A, B):

    num_of_batch = len(A)
    # honeybadgermpc.field.GFElement
    A_type = (type(A[0][0][0]) == GFElement)
    B_type = (type(B[0][0][0]) == GFElement)

    result = [[[0 for _ in range(len(A[0][0]))] for _ in range(len(A[0]))] for _ in range(num_of_batch)]
    # write matrices A  into files
    file_name = f"matrix_{ctx.myid}_A.input"
    file_path = f"sharedata/{file_name}"
    with open(file_path, "w") as f:
        for i in range(num_of_batch):      
            print(ctx.field.modulus, file=f)
            print(len(A[i]), file=f)
            print(len(A[i][0]), file=f)
            if(A_type):
                for ii in range(len(A[i])):
                    for jj in range(len(A[i][0])):
                        print(A[i][ii][jj].value, file=f)
            else:
                 for ii in range(len(A[i])):
                    for jj in range(len(A[i][0])):
                        print(A[i][ii][jj].v.value, file=f)
    # write matrices B into files
    file_name = f"matrix_{ctx.myid}_B.input"
    file_path = f"sharedata/{file_name}"
    with open(file_path, "w") as f:
        for i in range(num_of_batch):
            print(ctx.field.modulus, file=f)
            print(len(B[i]), file=f)
            print(len(B[i][0]), file=f)
            if(B_type):
                for ii in range(len(B[i])):
                    for jj in range(len(B[i][0])):
                        print(B[i][ii][jj].value, file=f)
            else:
                 for ii in range(len(B[i])):
                    for jj in range(len(B[i][0])):
                        print(B[i][ii][jj].v.value, file=f)  
    # do computation
    runcmd = f"./apps/tutorial/cpp/multi_matrix_sub {num_of_batch} {ctx.myid} {MODE}"
    await run_command_sync(runcmd)

    # # read C++ benchmark
    # file_name = f"benchmark_add.log"
    # file_path = f"sharedata/{file_name}"
    # cpp_time = 0;
    # with open(file_path, "r") as f:
    #     cpp_time = f.readline()
    # # logging.info(f"Batch_cpp_add, pure CPP time: {cpp_time}")
    # total_add_time = total_add_time + float(cpp_time)

    #load result from files
    file_name = f"matrix_{ctx.myid}_C.output"
    file_path = f"sharedata/{file_name}"
    with open(file_path, "r") as f:
        for i in range(num_of_batch):
            assert ctx.field.modulus == int(f.readline())
            row = int(f.readline())
            column = int(f.readline())
            if A_type and B_type:
                for r in range(row):
                    for c in range(column):
                        result[i][r][c] = ctx.field(int(f.readline()))
            else:
                for r in range(row):
                    for c in range(column):
                        result[i][r][c] = ctx.Share(int(f.readline()))                
    return result

async def batch_multi_matrices_multiply_with_precompute(ctx, M, R, R_inverse, super_triple, normal_triple):

    if len(M) < 2 or len(R) < 2 or len(M) != (len(R) - 1) or len(R_inverse) != len(R):
        return None
    # Notice: trick used here to save time, correct code should be len(normal_triple) != 9 * len(M)
    # if len(super_triple) != 4 * len(M) or len(normal_triple) != 3 * len(M):
    #     return None
    # tricks: same super triple used for all multiplications
    # multiply all R_iX_iR_(i+1)^(-1)
    t = await batch_beaver_mul_three_matrix_with_precomputation(ctx, R[:-1], M, R_inverse[1:], super_triple[0:4], normal_triple[0:3])
    start_open =  time.time()
    triples = await batch_matrix_open(ctx, t)
    stop_open =  time.time()
    logging.info(f"time for opening R_iX_iR_(i+1)^(-1): {start_open - stop_open}")

    startt = time.time()
    temp = len(triples)
    for it in range(int(math.log(len(M), 2))):
        result = await batch_cpp_matrix_mul(ctx, triples[::2], triples[1::2])
        triples = result
    stopt = time.time()
    last_time = stopt - startt
    logging.info(f"time for multiplying all R_iX_iR_(i+1)^(-1): {last_time}")
    result = triples[0]

    result = await batch_cpp_matrix_mul(ctx, [R_inverse[0]], [result]) 
    result = result[0]
    # Note: this can also be moved outside. TBD
    # A, B, C = generate_beaver_matrix_hack(ctx, len(M[0]), len(M[0]), len(M[0]))
    startf = time.time()
    result = await beaver_mul_matrix(ctx, result, R[-1], normal_triple[0], normal_triple[1], normal_triple[2])
    stopf = time.time()
    logging.info(f"time to compute [R_0^(-1)]result_of_mul[R_n]: {stopf - startf}")
    return result

# multiply n matrices by using only beaver matrice mul. Thies leads to a log(n) round complexity. n has to be power of 2.
# trick used here, normal_triple includes only 1 triple that is used for all multiplication. This is only for saving testing time. Don;t do it in reality.
async def state_of_art_mul(ctx, M, normal_triple):
    num_of_matrix = len(M)
    if num_of_matrix < 2:
        return None
    result = M
    # tricks: same super triple used for all multiplications
    for it in range(int(math.log(num_of_matrix, 2))):
        result = await batch_beaver_mul_matrix(ctx, result[::2], result[1::2], normal_triple[0], normal_triple[1], normal_triple[2])
    return result[0]


async def matrix_power(ctx, M, p, R, R_inv, normal_triple):
    triple = await batch_beaver_mul_three_matrix_with_precomputation(ctx, [R], [M], [R_inv], [], normal_triple)
    open_triple = await batch_matrix_open(ctx, triple)

    # tricks: same super triple used for all multiplications
    for it in range(int(math.log(p, 2))):
        open_triple = await batch_cpp_matrix_mul(ctx, open_triple, open_triple)

    result = await batch_cpp_matrix_mul(ctx, [R_inv], open_triple)   
    result = await beaver_mul_matrix(ctx, result[0], R, normal_triple[0], normal_triple[1], normal_triple[2])     
    return result



async def state_of_art_power(ctx, M, p, normal_triple):
    result = M
    # tricks: same super triple used for all multiplications
    for it in range(int(math.log(p, 2))):
        result = await beaver_mul_matrix(ctx, result, result, normal_triple[0], normal_triple[1], normal_triple[2])
    return result


def triple_generation_for_multi_matrix(ctx, k, n):
    super_triple = []
    normal_triple = []
    # hack used here
    for i in range(1):
        A,B,C,D = generate_beaver_triple_matrix_hack(ctx, k, k, k, k)
        super_triple.append(A)
        super_triple.append(B)
        super_triple.append(C)
        super_triple.append(D)
    # hack here
    for i in range(1):
        A, B, C = generate_beaver_matrix_hack(ctx, k, k, k)
        normal_triple.append(A)
        normal_triple.append(B)
        normal_triple.append(C)

    return super_triple, normal_triple



async def simple_matrix(ctx, **kwargs):
    global total_add_time
    global total_mul_time
    global total_communicate_time
    k = kwargs["k"]
    n = 1
    p = 256
    matrix_a = [[ctx.Share(i+j) for i in range(k)] for j in range(k)]
    # matrix_b = [[ctx.Share(5) for _ in range(k)] for _ in range(k)]
    await run_command_sync("chmod 777 ./apps/tutorial/cpp/multi_matrix_add")
    await run_command_sync("chmod 777 ./apps/tutorial/cpp/multi_matrix_sub")
    await run_command_sync("chmod 777 ./apps/tutorial/cpp/multi_matrix_mul")
    R, R_inverse = offline_multi_matrix_multiply(ctx, k, n)
    super_triple, normal_triple = triple_generation_for_multi_matrix(ctx, k, n)
    M = []
    for _ in range(n):
        M.append(matrix_a)
    start = time.time()
    # res = await batch_multi_matrices_multiply_with_precompute(ctx, M, R, R_inverse, super_triple, normal_triple)
    # res = await state_of_art_power(ctx, matrix_a, p, normal_triple)
    res = await matrix_power(ctx, matrix_a, p, R[0], R_inverse[0], normal_triple)
    stop = time.time()
    last_time = stop - start

    start2 = time.time()
    res = await state_of_art_power(ctx, matrix_a, p, normal_triple)
    stop2 = time.time()
    last_time2 = stop2 - start2
    # logging.info(f"total local add time:{total_add_time}")
    # logging.info(f"total local mul time:{total_mul_time}")
    # logging.info(f"pure communication time:{total_communicate_time}")
    logging.info(f"online time for our protocol:{last_time}")
    logging.info(f"online time for state of the art protocol:{last_time2}")
    return res

async def _run(peers, n, t, my_id, k):
    from honeybadgermpc.ipc import ProcessProgramRunner

    async with ProcessProgramRunner(peers, n, t, my_id, mpc_config) as runner:
        await runner.execute("0", simple_matrix, k=k)
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
    k = 10
    try:
    #     pp_elements = FakePreProcessedElements()
    #     if HbmpcConfig.my_id == 0:
            
    #         # pp_elements.generate_bits(k* 1000, HbmpcConfig.N, HbmpcConfig.t)
    #         # pp_elements.generate_rands(k * k * 5, HbmpcConfig.N, HbmpcConfig.t)
    #         pp_elements.generate_triples(k * k * 5, HbmpcConfig.N, HbmpcConfig.t)
    #         # pp_elements.generate_zeros(k * k, HbmpcConfig.N, HbmpcConfig.t)
    #         pp_elements.preprocessing_done()
    #     else:
    #         loop.run_until_complete(pp_elements.wait_for_preprocessing())

        loop.run_until_complete(
            _run(HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id, k)
        )
    finally:
        loop.close()
