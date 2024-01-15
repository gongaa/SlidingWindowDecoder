import numpy as np
from functools import reduce
from scipy.sparse import identity, hstack, kron, csr_matrix
from src.utils import row_echelon, rank, kernel, compute_code_distance, inverse, int2bin

# Some parts of the code are a refactored version of github.com/quantumgizmos/bp_osd

class css_code():
    # do as less row echelon form calculation as possible.
    def __init__(self, hx=np.array([[]]), hz=np.array([[]]), code_distance=np.nan, name=None, name_prefix="", check_css=False):

        self.hx = hx # hx pcm
        self.hz = hz # hz pcm

        self.lx = np.array([[]]) # x logicals
        self.lz = np.array([[]]) # z logicals

        self.N = np.nan # block length
        self.K = np.nan # code dimension
        self.D = code_distance # do not take this as the real code distance
        # TODO: use QDistRnd to get the distance
        # the quantum code distance is the minimum weight of all the affine codes
        # each of which is a coset code of a non-trivial logical op + stabilizers
        self.L = np.nan # max column weight
        self.Q = np.nan # max row weight

        _, nx = self.hx.shape
        _, nz = self.hz.shape

        assert nx == nz, "hx and hz should have equal number of columns!"
        assert nx != 0,  "number of variable nodes should not be zero!"
        if check_css: # For performance reason, default to False
            assert not np.any(hx @ hz.T % 2), "CSS constraint not satisfied"
        
        self.N = nx
        self.hx_perp, self.rank_hx, self.pivot_hx = kernel(hx) # orthogonal complement
        self.hz_perp, self.rank_hz, self.pivot_hz = kernel(hz)
        self.hx_basis = self.hx[self.pivot_hx] # same as calling row_basis(self.hx)
        self.hz_basis = self.hz[self.pivot_hz] # but saves one row echelon calculation
        self.K = self.N - self.rank_hx - self.rank_hz

        self.compute_ldpc_params()
        self.compute_logicals()
        if code_distance is np.nan:
            dx = compute_code_distance(self.hx_perp, is_pcm=False, is_basis=True)
            dz = compute_code_distance(self.hz_perp, is_pcm=False, is_basis=True)
            self.D = np.min([dx,dz]) # this is the distance of stabilizers, not the distance of the code

        self.name = f"{name_prefix}_n{self.N}_k{self.K}" if name is None else name

    def compute_ldpc_params(self):

        #column weights
        hx_l = np.max(np.sum(self.hx, axis=0))
        hz_l = np.max(np.sum(self.hz, axis=0))
        self.L = np.max([hx_l, hz_l]).astype(int)

        #row weights
        hx_q = np.max(np.sum(self.hx, axis=1))
        hz_q = np.max(np.sum(self.hz, axis=1))
        self.Q = np.max([hx_q, hz_q]).astype(int)

    def compute_logicals(self):

        def compute_lz(ker_hx, im_hzT):
            # lz logical operators
            # lz\in ker{hx} AND \notin Im(hz.T)
            # in the below we row reduce to find vectors in kx that are not in the image of hz.T.
            log_stack = np.vstack([im_hzT, ker_hx])
            pivots = row_echelon(log_stack.T)[3]
            log_op_indices = [i for i in range(im_hzT.shape[0], log_stack.shape[0]) if i in pivots]
            log_ops = log_stack[log_op_indices]
            return log_ops

        self.lx = compute_lz(self.hz_perp, self.hx_basis)
        self.lz = compute_lz(self.hx_perp, self.hz_basis)

        return self.lx, self.lz

    def canonical_logicals(self):
        temp = inverse(self.lx @ self.lz.T % 2)
        self.lx = temp @ self.lx % 2

def create_circulant_matrix(l, pows):
    h = np.zeros((l,l), dtype=int)
    for i in range(l):
        for c in pows:
            h[(i+c)%l, i] = 1
    return h


def create_generalized_bicycle_codes(l, a, b, name=None):
    A = create_circulant_matrix(l, a)
    B = create_circulant_matrix(l, b)
    hx = np.hstack((A, B))
    hz = np.hstack((B.T, A.T))
    return css_code(hx, hz, name=name, name_prefix="GB")


def hypergraph_product(h1, h2, name=None):
    m1, n1 = np.shape(h1)
    r1 = rank(h1)
    k1 = n1 - r1
    k1t = m1 - r1

    m2, n2 = np.shape(h2)
    r2 = rank(h2)
    k2 = n2 - r2
    k2t = m2 - r2

    #hgp code params
    N = n1 * n2 + m1 * m2
    K = k1 * k2 + k1t * k2t #number of logical qubits in hgp code

    #construct hx and hz
    h1 = csr_matrix(h1)
    hx1 = kron(h1, identity(n2, dtype=int))
    hx2 = kron(identity(m1, dtype=int), h2.T)
    hx = hstack([hx1, hx2]).toarray()

    h2 = csr_matrix(h2)
    hz1 = kron(identity(n1, dtype=int), h2)
    hz2 = kron(h1.T, identity(m2, dtype=int))
    hz = hstack([hz1, hz2]).toarray()
    return css_code(hx, hz, name=name, name_prefix="HP")

def hamming_code(rank):
    rank = int(rank)
    num_rows = (2**rank) - 1
    pcm = np.zeros((num_rows, rank), dtype=int)
    for i in range(0, num_rows):
        pcm[i] = int2bin(i+1, rank)
    return pcm.T

def rep_code(d):
    pcm = np.zeros((d-1, d), dtype=int)
    for i in range(d-1):
        pcm[i, i] = 1
        pcm[i, i+1] = 1
    return pcm

def create_surface_codes(n):
    # [n^2+(n-1)^2, 1, n] surface code
    h = rep_code(n)
    return hypergraph_product(h, h, f"Surface_n{n**2 + (n-1)**2}_k{1}_d{n}")

def set_pcm_row(n, pcm, row_idx, i, j):
    i1, j1 = (i+1) % n, (j+1) % n
    pcm[row_idx][i*n+j] = pcm[row_idx][i1*n+j1] = 1
    pcm[row_idx][i1*n+j] = pcm[row_idx][i*n+j1] = 1
     
def create_rotated_surface_codes(n, name=None):
    assert n % 2 == 1, "n should be odd"
    n2 = n*n
    m = (n2-1) // 2
    hx = np.zeros((m, n2), dtype=int)
    hz = np.zeros((m, n2), dtype=int)
    x_idx = 0
    z_idx = 0
   
    for i in range(n-1):
        for j in range(n-1):
            if (i+j) % 2 == 0: # Z check
                set_pcm_row(n, hz, z_idx, i, j)
                z_idx += 1
            else: # X check
                set_pcm_row(n, hx, x_idx, i, j)
                x_idx += 1    

    # upper and lower edge, weight-2 X checks
    for j in range(n-1):
        if j % 2 == 0: # upper 
            hx[x_idx][j] = hx[x_idx][j+1] = 1
        else:
            hx[x_idx][(n-1)*n+j] = hx[x_idx][(n-1)*n+(j+1)] = 1
        x_idx += 1
        
    # left and right edge, weight-2 Z checks
    for i in range(n-1):
        if i % 2 == 0: # right
            hz[z_idx][i*n+(n-1)] = hz[z_idx][(i+1)*n+(n-1)] = 1
        else:
            hz[z_idx][i*n] = hz[z_idx][(i+1)*n] = 1
        z_idx += 1
    
    return css_code(hx, hz, name=name, name_prefix="Rotated_Surface")

def create_checkerboard_toric_codes(n, name=None):
    assert n % 2 == 0, "n should be even"
    n2 = n*n
    m = (n2) // 2
    hx = np.zeros((m, n2), dtype=int)
    hz = np.zeros((m, n2), dtype=int)
    x_idx = 0
    z_idx = 0
    
    for i in range(n):
        for j in range(n):
            if (i+j) % 2 == 0: # Z check
                set_pcm_row(n, hz, z_idx, i, j)
                z_idx += 1
            else:
                set_pcm_row(n, hx, x_idx, i, j)
                x_idx += 1
    
    return css_code(hx, hz, name=name, name_prefix="Toric")   
  
def create_QC_GHP_codes(l, a, b, name=None):
    # quasi-cyclic generalized hypergraph product codes
    m, n = a.shape
    block_list = []
    for row in a:
        temp = []
        for s in row:
            if s >= 0:
                temp.append(create_circulant_matrix(l, [s]))
            else:
                temp.append(np.zeros((l,l), dtype=int))
        block_list.append(temp)
    A = np.block(block_list) # ml * nl

    temp_b = create_circulant_matrix(l, b)
    B = np.kron(np.identity(m, dtype=int), temp_b)
    hx = np.hstack((A, B))
    B_T = np.kron(np.identity(n, dtype=int), temp_b.T)
    hz = np.hstack((B_T, A.T))
    return css_code(hx, hz, name=name, name_prefix=f"GHP")

def create_cyclic_permuting_matrix(n, shifts):
    A = np.full((n,n), -1, dtype=int)
    for i, s in enumerate(shifts):
        for j in range(n):
            A[j, (j-i)%n] = s
    return A
        
def create_bivariate_QC_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows, name=None):
    S_l=create_circulant_matrix(l, [-1])
    S_m=create_circulant_matrix(m, [-1])
    x = kron(S_l, identity(m, dtype=int))
    y = kron(identity(l, dtype=int), S_m)
    A_list = [x**p for p in A_x_pows] + [y**p for p in A_y_pows]
    B_list = [y**p for p in B_y_pows] + [x**p for p in B_x_pows] 
    A = reduce(lambda x,y: x+y, A_list).toarray()
    B = reduce(lambda x,y: x+y, B_list).toarray()
    hx = np.hstack((A, B))
    hz = np.hstack((B.T, A.T))
    return css_code(hx, hz, name=name, name_prefix="IBM"), A_list, B_list

# For reading in overcomplete check matrices
def readAlist(directory):
    '''
    Reads in a parity check matrix (pcm) in A-list format from text file. returns the pcm in form of a numpy array with 0/1 bits as float64.
    '''
    alist_raw = []
    with open(directory, "r") as f:
        lines = f.readlines()
        for line in lines:
            # remove trailing newline \n and split at spaces:
            line = line.rstrip().split(" ")
            # map string to int:
            line = list(map(int, line))
            alist_raw.append(line)
    alist_numpy = alistToNumpy(alist_raw)
    alist_numpy = alist_numpy.astype(int)
    return alist_numpy


def alistToNumpy(lines):
    '''Converts a parity-check matrix in AList format to a 0/1 numpy array'''
    nCols, nRows = lines[0]
    if len(lines[2]) == nCols and len(lines[3]) == nRows:
        startIndex = 4
    else:
        startIndex = 2
    matrix = np.zeros((nRows, nCols), dtype=float)
    for col, nonzeros in enumerate(lines[startIndex:startIndex + nCols]):
        for rowIndex in nonzeros:
            if rowIndex != 0:
                matrix[rowIndex - 1, col] = 1
    return matrix

