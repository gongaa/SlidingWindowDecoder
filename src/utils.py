#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions and layers for the FEC package."""

import numpy as np
from collections import deque

def bin2int(arr):
    """Convert binary array to integer.

    For example ``arr`` = `[1, 0, 1]` is converted to `5`.

    Input
    -----
        arr: int or float
            An iterable that yields 0's and 1's.

    Output
    -----
        : int
            Integer representation of ``arr``.

    """
    if len(arr) == 0: return None
    return int(''.join([str(x) for x in arr]), 2)

def int2bin(num, len_):
    """
    Convert ``num`` of int type to list of length ``len_`` with 0's and 1's.
    ``num`` and ``len_`` have to non-negative.

    For e.g., ``num`` = `5`; `int2bin(num`, ``len_`` =4) = `[0, 1, 0, 1]`.

    For e.g., ``num`` = `12`; `int2bin(num`, ``len_`` =3) = `[1, 0, 0]`.

    Input
    -----
        num: int
            An integer to be converted into binary representation.

        len_: int
            An integer defining the length of the desired output.

    Output
    -----
        : list of int
            Binary representation of ``num`` of length ``len_``.
    """
    assert num >= 0,  "Input integer should be non-negative"
    assert len_ >= 0,  "width should be non-negative"

    bin_ = format(num, f'0{len_}b')
    binary_vals = [int(x) for x in bin_[-len_:]] if len_ else []
    return binary_vals

def alist2mat(alist, verbose=True):
    # pylint: disable=line-too-long
    r"""Convert `alist` [MacKay]_ code definition to `full` parity-check matrix.

    Many code examples can be found in [UniKL]_.

    About `alist` (see [MacKay]_ for details):

        - `1.` Row defines parity-check matrix dimension `m x n`
        - `2.` Row defines int with `max_CN_degree`, `max_VN_degree`
        - `3.` Row defines VN degree of all `n` column
        - `4.` Row defines CN degree of all `m` rows
        - Next `n` rows contain non-zero entries of each column (can be zero padded at the end)
        - Next `m` rows contain non-zero entries of each row.

    Input
    -----
    alist: list
        Nested list in `alist`-format [MacKay]_.

    verbose: bool
        Defaults to True. If True, the code parameters are printed.

    Output
    ------
    (pcm, k, n, coderate):
        Tuple:

    pcm: ndarray
        NumPy array of shape `[n-k, n]` containing the parity-check matrix.

    k: int
        Number of information bits.

    n: int
        Number of codewords bits.

    coderate: float
        Coderate of the code.

    Note
    ----
        Use :class:`~sionna.fec.utils.load_alist` to import alist from a
        textfile.

        For example, the following code snippet will import an alist from a file called ``filename``:

        .. code-block:: python

            al = load_alist(path = filename)
            pcm, k, n, coderate = alist2mat(al)
    """

    assert len(alist)>4, "Invalid alist format."

    n = alist[0][0]
    m = alist[0][1]
    v_max = alist[1][0]
    c_max = alist[1][1]
    k = n - m
    coderate = k / n

    vn_profile = alist[2]
    cn_profile = alist[3]

    # plausibility checks
    assert np.sum(vn_profile)==np.sum(cn_profile), "Invalid alist format."
    assert np.max(vn_profile)==v_max, "Invalid alist format."
    assert np.max(cn_profile)==c_max, "Invalid alist format."

    if len(alist)==len(vn_profile)+4:
        print("Note: .alist does not contain (redundant) CN perspective.")
        print("Recovering parity-check matrix from VN only.")
        print("Please verify the correctness of the results manually.")
        vn_only = True
    else:
        assert len(alist)==len(vn_profile) + len(cn_profile) + 4, \
                                                "Invalid alist format."
        vn_only = False

    pcm = np.zeros((m,n))
    num_edges = 0 # count number of edges

    for idx_v in range(n):
        for idx_i in range(vn_profile[idx_v]):
            # first 4 rows of alist contain meta information
            idx_c = alist[4+idx_v][idx_i]-1 # "-1" as this is python
            pcm[idx_c, idx_v] = 1
            num_edges += 1 # count number of edges (=each non-zero entry)

    # validate results from CN perspective
    if not vn_only:
        for idx_c in range(m):
            for idx_i in range(cn_profile[idx_c]):
                # first 4 rows of alist contain meta information
                # follwing n rows contained VN perspective
                idx_v = alist[4+n+idx_c][idx_i]-1 # "-1" as this is python
                assert pcm[idx_c, idx_v]==1 # entry must already exist

    if verbose:
        print("Number of variable nodes (columns): ", n)
        print("Number of check nodes (rows): ", m)
        print("Number of information bits per cw: ", k)
        print("Number edges: ", num_edges)
        print("Max. VN degree: ", v_max)
        print("Max. CN degree: ", c_max)
        print("VN degree: ", vn_profile)
        print("CN degree: ", cn_profile)

    return pcm, k, n, coderate

def load_alist(path):
    """Read `alist`-file [MacKay]_ and return nested list describing the
    parity-check matrix of a code.

    Many code examples can be found in [UniKL]_.

    Input
    -----
    path:str
        Path to file to be loaded.

    Output
    ------
    alist: list
        A nested list containing the imported alist data.
    """

    alist = []
    with open(path, "r") as reader: # pylint: disable=unspecified-encoding
        # read list line by line (different length)
        for line in reader:
            l = []
            # append all entries
            for word in line.split():
                l.append(int(word))
            if l: # ignore empty lines
                alist.append(l)

    return alist

def make_systematic(mat, is_pcm=False):
    r"""Bring binary matrix in its systematic form.

    Input
    -----
    mat : ndarray
        Binary matrix to be transformed to systematic form of shape `[k, n]`.

    is_pcm: bool
        Defaults to False. If true, ``mat`` is interpreted as parity-check
        matrix and, thus, the last k columns will be the identity part.

    Output
    ------
    mat_sys: ndarray
        Binary matrix in systematic form, i.e., the first `k` columns equal the
        identity matrix (or last `k` if ``is_pcm`` is True).

    column_swaps: list of int tuples
        A list of integer tuples that describes the swapped columns (in the
        order of execution).

    Note
    ----
    This algorithm (potentially) swaps columns of the input matrix. Thus, the
    resulting systematic matrix (potentially) relates to a permuted version of
    the code, this is defined by the returned list ``column_swap``.
    Note that, the inverse permutation must be applied in the inverse list
    order (in case specific columns are swapped multiple times).

    If a parity-check matrix is passed as input (i.e., ``is_pcm`` is True), the
    identity part will be re-arranged to the last columns."""

    m = mat.shape[0]
    n = mat.shape[1]

    assert m<=n, "Invalid matrix dimensions."

    # check for all-zero columns (=unchecked nodes)
    if is_pcm:
        c_node_deg = np.sum(mat, axis=0)
        if np.any(c_node_deg==0):
            warnings.warn("All-zero column in parity-check matrix detected. " \
                "It seems as if the code contains unprotected nodes.")

    mat = np.copy(mat)
    column_swaps = [] # store all column swaps

    # convert to bool for faster arithmetics
    mat = mat.astype(bool)

    # bring in upper triangular form
    for idx_c in range(m):
        success = False
        # step 1: find next leading "1"
        for idx_r in range(idx_c,m):
            # skip if entry is "0"
            if mat[idx_r, idx_c]:
                mat[[idx_c, idx_r]] = mat[[idx_r, idx_c]] # swap rows
                success = True
                break

        # Could not find "1"-entry for column idx_c
        # => swap with columns from non-sys part
        # The task is to find a column with index idx_cc that has a "1" at
        # row idx_c
        if not success:
            for idx_cc in range(m, n):
                if mat[idx_c, idx_cc]:
                    # swap columns
                    mat[:,[idx_c, idx_cc]] = mat[:,[idx_cc, idx_c]]
                    column_swaps.append([idx_c, idx_cc])
                    success=True
                    break

        if not success:
            raise ValueError("Could not succeed; mat is not full rank?")

        # we can now assume a leading "1" at row idx_c
        for idx_r in range(idx_c+1, m):
            if mat[idx_r, idx_c]:
                mat[idx_r,:] ^= mat[idx_c,:] # bin. add of row idx_c to idx_r

    # remove upper triangle part in inverse order
    for idx_c in range(m-1, -1, -1):
        for idx_r in range(idx_c-1, -1, -1):
            if mat[idx_r, idx_c]:
                mat[idx_r,:] ^= mat[idx_c,:] # bin. add of row idx_c to idx_r

    # verify results
    assert np.array_equal(mat[:,:m], np.eye(m)), \
                            "Internal error, could not find systematic matrix."

    # bring identity part to end of matrix if parity-check matrix is provided
    if is_pcm:
        im = np.copy(mat[:,:m])
        mat[:,:m] = mat[:,-m:]
        mat[:,-m:] = im
        # and track column swaps
        for idx in range(m):
            column_swaps.append([idx, n-m+idx])

    # return integer array
    mat = mat.astype(int)
    return mat, column_swaps

###########################################################
# Functions adapted from the ldpc package
###########################################################

def row_echelon(mat, reduced=False):
    r"""Converts a binary matrix to (reduced) row echelon form via Gaussian Elimination, 
    also works for rank-deficient matrix. Unlike the make_systematic method,
    no column swaps will be performed.

    Input 
    ----------
    mat : ndarry
        A binary matrix in numpy.ndarray format.
    reduced: bool
        Defaults to False. If true, the reduced row echelon form is returned. 
    
    Output
    -------
    row_ech_form: ndarray
        The row echelon form of input matrix.
    rank: int
        The rank of the matrix.
    transform: ndarray
        The transformation matrix such that (transform_matrix@matrix)=row_ech_form
    pivot_cols: list
        List of the indices of pivot num_cols found during Gaussian elimination
    """

    m, n = np.shape(mat)
    # Don't do "m<=n" check, allow over-complete matrices
    mat = np.copy(mat)
    # Convert to bool for faster arithmetics
    mat = mat.astype(bool)
    transform = np.identity(m).astype(bool)
    pivot_row = 0
    pivot_cols = []

    # Allow all-zero column. Row operations won't induce all-zero columns, if they are not present originally.
    # The make_systematic method will swap all-zero columns with later non-all-zero columns.
    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(n):
        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if not mat[pivot_row, col]:
            # Find a row with a 1 in this column
            swap_row_index = pivot_row + np.argmax(mat[pivot_row:m, col])
            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if mat[swap_row_index, col]:
                # Swap rows
                mat[[swap_row_index, pivot_row]] = mat[[pivot_row, swap_row_index]]
                # Transformation matrix update to reflect this row swap
                transform[[swap_row_index, pivot_row]] = transform[[pivot_row, swap_row_index]]

        if mat[pivot_row, col]: # will evaluate to True if this column is not all-zero
            if not reduced: # clean entries below the pivot 
                elimination_range = [k for k in range(pivot_row + 1, m)]
            else:           # clean entries above and below the pivot
                elimination_range = [k for k in range(m) if k != pivot_row]
            for idx_r in elimination_range:
                if mat[idx_r, col]:    
                    mat[idx_r] ^= mat[pivot_row]
                    transform[idx_r] ^= transform[pivot_row]
            pivot_row += 1
            pivot_cols.append(col)

        if pivot_row >= m: # no more rows to search
            break

    rank = pivot_row
    row_ech_form = mat.astype(int)

    return [row_ech_form, rank, transform.astype(int), pivot_cols]

def rank(mat):
    r"""Returns the rank of a binary matrix

    Input 
    ----------
    mat: ndarray
        A binary matrix in numpy.ndarray format

    Output
    -------
    int
        The rank of the matrix"""
    return row_echelon(mat)[1]

def kernel(mat):
    r"""Computes the kernel of the matrix M.
    All vectors x in the kernel of M satisfy the following condition::

        Mx=0 \forall x \in ker(M)

    Input 
    ----------
    mat: ndarray
        A binary matrix in numpy.ndarray format.
    
    Output
    -------
    ker: ndarray
        A binary matrix which is the kernel of the input binary matrix.

    rank: int
        Rank of transposed mat, which is the same as the rank of mat.

    pivot_cols: list
        List of the indices of pivot of the transposed mat. Can be used in row_basis.
    
    Note
    -----
    Why does this work?

    The transformation matrix, P, transforms the matrix M into row echelon form, ReM::

        P@M=ReM=[A,0]^T,
    
    where the width of A is equal to the rank. This means the bottom n-k rows of P
    must produce a zero vector when applied to M. For a more formal definition see
    the Rank-Nullity theorem.
    """

    transpose = mat.T
    m, _ = transpose.shape
    _, rank, transform, pivot_cols = row_echelon(transpose)
    ker = transform[rank:m]
    return ker, rank, pivot_cols

def row_basis(mat):
    r"""Outputs a basis for the rows of the matrix.

    Input
    ----------
    mat: ndarray
        The input matrix.

    Output
    -------
    basis: ndarray
        A numpy.ndarray matrix where each row is a basis element."""
    return mat[row_echelon(mat.T)[3]]

def compute_code_distance(mat, is_pcm=True, is_basis=False):
    r'''Computes the distance of the linear code given by the input parity check / generator matrix. 
    The code distance is given by the minimum weight of a nonzero codeword.

    Note
    ----
    The runtime of this function scales exponentially with the block size. In practice, computing the code distance of codes with block lengths greater than ~10 will be very slow.

    Parameters
    ----------
    mat: ndarray
        The parity check matrix
    
    is_pcm: bool
        Defaults to True. If false, mat is interpreted as a generator matrix.
    
    Returns
    -------
    int
        The code distance
    '''
    gen = mat
    if is_pcm:
        gen = kernel(mat)
    if len(gen)==0: return np.inf # infinite code distance
    cw = gen
    if not is_basis:
        cw = row_basis(gen) # nonzero codewords
    return np.min(np.sum(cw, axis=1))

def inverse(mat):
    r"""Computes the left inverse of a full-rank matrix.

    Input
    ----------
    matrix: ndarray
        The binary matrix to be inverted in numpy.ndarray format. This matrix must either be
        square full-rank or rectangular with full-column rank.

    Output
    -------
    inverse: ndarray
        The inverted binary matrix
    
    Note
    -----
    The `left inverse' is computed when the number of rows in the matrix
    exceeds the matrix rank. The left inverse is defined as follows::

        Inverse(M.T@M)@M.T

    We can make a further simplification by noting that the row echelon form matrix
    with full column rank has the form::

        row_echelon_form=P@M=vstack[I,A]

    In this case the left inverse simplifies to::

        Inverse(M^T@P^T@P@M)@M^T@P^T@P=M^T@P^T@P=row_echelon_form.T@P"""

    m, n = mat.shape
    reduced_row_ech, rank, transform, _ = row_echelon(mat, reduced=True)
    if m == n and rank == m:
        return transform
    # compute the left-inverse
    elif m > rank and n == rank:  # left inverse
        return reduced_row_ech.T @ transform % 2
    else:
        raise ValueError("This matrix is not invertible. Please provide either a full-rank square\
        matrix or a rectangular matrix with full column rank.")

def hopcroft_karp(adj, U, V):
    """
    Hopcroft-Karp maximum matching for bipartite graphs.

    Args:
        adj (dict): adjacency list from U to list of neighbors in V
        U (iterable): left vertex set
        V (iterable): right vertex set

    Returns:
        dict: matching as a map u->v for matched pairs
    """
    INF = float('inf')
    pair_U = {u: None for u in U}
    pair_V = {v: None for v in V}
    dist = {}

    def bfs():
        queue = deque()
        for u in U:
            if pair_U[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        dist[None] = INF

        while queue:
            u = queue.popleft()
            if dist[u] < dist[None]:
                for v in adj.get(u, []):
                    pu = pair_V[v]
                    if pu is None:
                        dist[None] = dist[u] + 1
                    elif dist[pu] == INF:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
        return dist[None] != INF

    def dfs(u):
        if u is not None:
            for v in adj.get(u, []):
                pu = pair_V[v]
                if pu is None or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_U[u] = v
                    pair_V[v] = u
                    return True
            dist[u] = INF
            return False
        return True

    matching = 0
    while bfs():
        for u in U:
            if pair_U[u] is None and dfs(u):
                matching += 1
    return {u: pair_U[u] for u in U if pair_U[u] is not None}


def edge_coloring_bipartite(adj_mat):
    """
    Edge-color a bipartite graph using Δ colors by iteratively extracting maximum matchings.

    Args:
        U (iterable): vertices in left partition
        V (iterable): vertices in right partition
        edges (list of tuple): list of (u,v) edges with u in U and v in V

    Returns:
        dict: mapping (u,v) to color index (1..Δ)
        int: the number of colors used (= max degree)
    """
    # Build adjacency list from adjacency matrix
    num_row, num_col = adj_mat.shape
    U, V = list(range(num_row)), list(range(num_col))
    # Build adjacency list copy
    adj = {u: [] for u in U}
    for u, v in zip(*adj_mat.nonzero()):
        adj[u].append(v)

    # Compute maximum degree Δ
    Delta = max(np.max(adj_mat.sum(axis=0)), np.max(adj_mat.sum(axis=1)))

    # Coloring by repeated matchings
    color = {}
    current_adj = {u: list(neighs) for u, neighs in adj.items()}
    num_colors = 0
    color_dict = {}
    for i in range(Delta):
        color_dict[i] = []
    while any(current_adj[u] for u in U):
        num_colors += 1
        # find a maximum matching
        M = hopcroft_karp(current_adj, U, V)
        # assign this matching the new color
        for u, v in M.items():
            color[(u, v)] = num_colors
            color_dict[num_colors-1].append((u,v))
            current_adj[u].remove(v) # remove colored edge
    assert num_colors <= Delta
    return color_dict, num_colors