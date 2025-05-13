import numpy as np
import itertools
import stim
import matplotlib.pyplot as plt
from functools import reduce
from .codes_q import gcd, poly_divmod, coeff2poly
from .utils import inverse, edge_coloring_bipartite

def build_SHYPS_circuit(r, p, num_repeat, z_basis=True, use_both=False):
    n_r = 2**r - 1
    # Primitive polynomial h(x)=1+x^a+x^b\in\mathbb{F}_2[x]/(x^{n_r}-1)
    # such that  gcd(h(x),x^{n_r}-1) is a primitive polynomial of degree r
    if r == 3:
        primitive_poly = [0,2,3] # h(x)=x^0+x^2+x^3
    elif r == 4:
        primitive_poly = [0,3,4] # h(x)=x^0+x^3+x^4
    elif r == 5:
        primitive_poly = [0,2,5] # h(x)=x^0+x^2+x^5
    else:
        print(f"Unsupported r={r}, please find primitive polynomial yourself")
    assert gcd([0,n_r], primitive_poly) == primitive_poly # check h(x) indeed divides (x^{n_r}-1)
    primitive_poly = coeff2poly(primitive_poly)[::-1] # list of coeff, in increasing order of degree
    # Define overcomplete PCM for classical simplex code
    H_first_row = np.zeros(n_r, dtype=int)
    H_first_row[:len(primitive_poly)] = primitive_poly
    H = np.array([np.roll(H_first_row, i) for i in range(n_r)]) # shape n_r by n_r
    print(H)
    generator_poly, _ = poly_divmod(coeff2poly([0,n_r])[::-1], primitive_poly, 2) # g(x) = (x^{n_r}-1) / h(x)
    G_first_row = np.zeros(n_r, dtype=int)
    G_first_row[:len(generator_poly)] = generator_poly
    G = np.array([np.roll(G_first_row, i) for i in range(r)]) # shape r by n_r
    print(G)
    assert not np.any(G @ H % 2) # GH=0, HG=0

    identity = np.identity(n_r, dtype=int)
    S_X = np.kron(H.T, G) # X stabilizers
    gauge_X = np.kron(H.T, identity) # X gauge operators
    aggregate_X = np.kron(identity, G)
    # to aggregate gauge operators into stabilizer
    # S_X = H \otimes G = (I \otimes G) gauge_X
    S_Z = np.kron(G, H.T) # Z stabilizers
    assert np.array_equal(S_Z.T, np.kron(G.T, H)) # (A \otimes B)^T = A^T \otimes B^T
    gauge_Z = np.kron(identity, H.T) # Z gauge operators
    aggregate_Z = np.kron(G, identity)

    assert not np.any(S_X @ S_Z.T % 2) # X and Z stabilizers commute
    assert not np.any(gauge_X @ S_Z.T % 2) # gauge X operators commute with Z stabilizers
    assert not np.any(S_X @ gauge_Z.T % 2) # gauge Z operators commute with X stabilizers

    # to define logical operators, first get the pivot matrix P (shape r by n_r)
    # such that P G^T = I
    P = inverse(G.T)
    # print(P)
    L_X = np.kron(P, G) # X logicals
    L_Z = np.kron(G, P) # Z logicals
    assert not np.any(gauge_X @ L_Z.T % 2) # gauge X operators commute with Z logicals
    assert not np.any(L_X @ gauge_Z.T % 2) # gauge Z operators commute with X logicals

    N = n_r ** 2 # number of data qubits, also number of X and Z gauge operators

    color_dict_gauge_X, num_colors_X = edge_coloring_bipartite(gauge_X)
    color_dict_gauge_Z, num_colors_Z = edge_coloring_bipartite(gauge_Z)
    assert num_colors_X == 3
    assert num_colors_Z == 3

    for color in range(3):
        print(f"color={color}, #edges: {len(color_dict_gauge_Z[color])}")
    X_gauge_offset = 0
    data_offset = N # there are N X gauge operators
    Z_gauge_offset = 2*N

    # first round (encoding round) detector circuit, only put on one basis, no previous round to XOR with
    detector_circuit_str = ""
    for row in (aggregate_Z if z_basis else aggregate_X):
        temp = "DETECTOR "
        for i in row.nonzero()[0]:
            temp += f"rec[{-N+i}] "
        detector_circuit_str += f"{temp}\n"
    detector_circuit = stim.Circuit(detector_circuit_str)

    X_detector_circuit_str = ""
    for row in aggregate_X:
        temp = "DETECTOR "
        for i in row.nonzero()[0]:
            temp += f"rec[{-N+i}] rec[{-3*N+i}] "
        X_detector_circuit_str += f"{temp}\n"
    X_detector_circuit = stim.Circuit(X_detector_circuit_str)

    Z_detector_circuit_str = ""
    for row in aggregate_Z:
        temp = "DETECTOR "
        for i in row.nonzero()[0]:
            temp += f"rec[{-N+i}] rec[{-3*N+i}] "
        Z_detector_circuit_str += f"{temp}\n"
    Z_detector_circuit = stim.Circuit(Z_detector_circuit_str)

    def append_block(circuit, repeat=False):
        if repeat: # not encoding round
            for i in range(N):
                circuit.append("X_ERROR", Z_gauge_offset + i, p)
                circuit.append("Z_ERROR", X_gauge_offset + i, p)
                circuit.append("DEPOLARIZE1", data_offset + i, p)
            circuit.append("TICK")

        for color in range(num_colors_Z):
            for Z_gauge_idx, data_idx in color_dict_gauge_Z[color]:
                circuit.append("CNOT", [data_offset + data_idx, Z_gauge_offset + Z_gauge_idx])
                circuit.append("DEPOLARIZE2", [data_offset + data_idx, Z_gauge_offset + Z_gauge_idx], p)
            circuit.append("TICK")

        # measure Z gauge operators
        for i in range(N):
            circuit.append("X_ERROR", Z_gauge_offset + i, p)
            circuit.append("M", Z_gauge_offset + i)
        if z_basis:
            circuit += (Z_detector_circuit if repeat else detector_circuit)
        # initialize X gauge operators
        for i in range(N):
            circuit.append("RX", X_gauge_offset + i)
            circuit.append("Z_ERROR", X_gauge_offset + i, p)
        circuit.append("TICK")

        for color in range(num_colors_X):
            for X_gauge_idx, data_idx in color_dict_gauge_X[color]:
            # for data_idx, X_gauge_idx in color_dict_gauge_X[color]:
                circuit.append("CNOT", [X_gauge_offset + X_gauge_idx, data_offset + data_idx])
                circuit.append("DEPOLARIZE2", [X_gauge_offset + X_gauge_idx, data_offset + data_idx], p)
            
            circuit.append("TICK")
        
        # measure X gauge operators
        for i in range(N):
            circuit.append("Z_ERROR", X_gauge_offset + i, p)
            circuit.append("MX", X_gauge_offset + i)
        if not z_basis:
            circuit += (X_detector_circuit if repeat else detector_circuit)
        # initialize Z gauge operators
        for i in range(N):
            circuit.append("R", Z_gauge_offset + i)
            circuit.append("X_ERROR", Z_gauge_offset + i, p)
        circuit.append("TICK")



    circuit = stim.Circuit()
    for i in range(N):
        circuit.append("RX", X_gauge_offset + i)
        circuit.append("Z_ERROR", X_gauge_offset + i, p)
        circuit.append("R", Z_gauge_offset + i)
        circuit.append("X_ERROR", Z_gauge_offset + i, p)
    for i in range(N):
        circuit.append("R" if z_basis else "RX", data_offset + i)
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", data_offset + i, p)

    # begin round tick
    circuit.append("TICK")
    append_block(circuit, repeat=False) # encoding round

    rep_circuit = stim.Circuit()
    append_block(rep_circuit, repeat=True)
    circuit += (num_repeat-1) * rep_circuit

    for i in range(N):
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", data_offset + i, p)
        circuit.append("M" if z_basis else "MX", data_offset + i)

    pcm = S_Z if z_basis else S_X
    aggregate_matrix = aggregate_Z if z_basis else aggregate_X
    logical_pcm = L_Z if z_basis else L_X
    stab_detector_circuit_str = ""
    row_idx = 0
    for row in pcm:
        det_str = "DETECTOR "
        for data_idx in row.nonzero()[0]:
            det_str += f"rec[{-N+data_idx}] "
        for gauge_idx in aggregate_matrix[row_idx].nonzero()[0]:
            det_str += f"rec[{-(3 if z_basis else 2)*N+gauge_idx}] "
        stab_detector_circuit_str += f"{det_str}\n"
        row_idx += 1
    circuit += stim.Circuit(stab_detector_circuit_str)

    log_detector_circuit_str = "" # logical operators
    row_idx = 0
    for row in logical_pcm:
        obs_str = f"OBSERVABLE_INCLUDE({row_idx}) "
        for data_idx in row.nonzero()[0]:
            obs_str += f"rec[{-N+data_idx}] "
        log_detector_circuit_str += f"{obs_str}\n"
        row_idx += 1
    circuit += stim.Circuit(log_detector_circuit_str)

    return circuit