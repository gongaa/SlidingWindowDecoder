import stim
print(stim.__version__)

import matplotlib.pyplot as plt
import numpy as np
import math

from ldpc import BpDecoder, BpOsdDecoder
import time
from src.utils import rank
from src.codes_q import create_bivariate_bicycle_codes, create_circulant_matrix
from src.build_circuit import build_circuit, dem_to_check_matrices
from src import bpgdg_decoder
# import itt # install see https://github.com/oleksandr-pavlyk/itt-python, you also need to install VTune

hard_samples = []

decoding_time = []
def sliding_window_decoder(N, p=0.003, num_repeat=12, num_shots=10000, max_iter=200, W=3, F=1, z_basis=True, 
                           noisy_prior=None, method=1, plot=False, low_error_mode=False,
                           max_step=25, max_iter_per_step=6, max_tree_depth=3, max_side_depth=10, max_side_branch_step=10,
                           last_win_gdg_factor=1.0, last_win_bp_factor=1.0):
    
    if N == 72:
        code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1,2], [1,2], [3]) # 72
    elif N == 90:
        code, A_list, B_list = create_bivariate_bicycle_codes(15, 3, [9], [1,2], [2,7], [0]) # 90
    elif N == 108:
        code, A_list, B_list = create_bivariate_bicycle_codes(9, 6, [3], [1,2], [1,2], [3]) # 108
    elif N == 144:
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3]) # 144
    elif N == 288:
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 12, [3], [2,7], [1,2], [3]) # 288
    elif N == 360:
        code, A_list, B_list = create_bivariate_bicycle_codes(30, 6, [9], [1,2], [25,26], [3]) # 360
    elif N == 756:
        code, A_list, B_list = create_bivariate_bicycle_codes(21,18, [3], [10,17], [3,19], [5]) # 756
    else:
        print("unsupported N")
        return

    circuit = build_circuit(code, A_list, B_list, p, num_repeat, z_basis=z_basis)
    dem = circuit.detector_error_model()
    chk, obs, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)
    num_row, num_col = chk.shape
    n = code.N
    n_half = n//2

    lower_bounds = []
    upper_bounds = []
    i = 0
    while i < num_row:
        lower_bounds.append(i)
        upper_bounds.append(i+n_half)
        if i+n > num_row:
            break
        lower_bounds.append(i)
        upper_bounds.append(i+n)
        i += n_half

    region_dict = {}
    for i, (l,u) in enumerate(zip(lower_bounds, upper_bounds)):
        region_dict[(l,u)] = i

    region_cols = [[] for _ in range(len(region_dict))]

    for i in range(num_col):
        nnz_col = np.nonzero(chk[:,i])[0]
        l = nnz_col.min() // n_half * n_half
        u = (nnz_col.max() // n_half + 1) * n_half
        region_cols[region_dict[(l,u)]].append(i)  

    chk = np.concatenate([chk[:,col].toarray() for col in region_cols], axis=1)
    obs = np.concatenate([obs[:,col].toarray() for col in region_cols], axis=1)
    priors = np.concatenate([priors[col] for col in region_cols])

    anchors = []
    j = 0
    for i in range(num_col):
        nnz_col = np.nonzero(chk[:,i])[0]
        if (nnz_col.min() >= j):
            anchors.append((j, i))
            j += n_half
    anchors.append((num_row, num_col))
    
    if noisy_prior is None and method != 0:
        b = anchors[W]
        c = anchors[W-1]
        if method == 1:
            c = (c[0], c[1]+n_half*3) # try also this for x basis
        noisy_prior = np.sum(chk[c[0]:b[0],c[1]:b[1]] * priors[c[1]:b[1]], axis=1)
        print("prior for noisy syndrome", noisy_prior[0])

    if method != 0:
        noisy_syndrome_priors = np.ones(n_half) * noisy_prior
    
    num_win = math.ceil((len(anchors)-W+F-1) / F)
    chk_submats = []
    prior_subvecs = []
    if plot:
        fig, ax = plt.subplots(num_win, 1)
    top_left = 0

    for i in range(num_win):
        a = anchors[top_left]
        bottom_right = min(top_left + W, len(anchors)-1)
        b = anchors[bottom_right]

        if i != num_win-1 and method != 0: # not the last round
            c = anchors[top_left + W - 1]
            if method == 1:
                c = (c[0], c[1]+n_half*3) # try also this for x basis
            noisy_syndrome = np.zeros((n_half*W,n_half))
            noisy_syndrome[-n_half:,:] = np.eye(n_half)# * noisy_syndrome_prior
            mat = chk[a[0]:b[0],a[1]:c[1]]
            mat = np.hstack((mat, noisy_syndrome))
            prior = priors[a[1]:c[1]]
            prior = np.concatenate((prior, noisy_syndrome_priors))
        else: # method==0 or last round
            mat = chk[a[0]:b[0],a[1]:b[1]]
            prior = priors[a[1]:b[1]]
        chk_submats.append(mat)
        prior_subvecs.append(prior)
        if plot:
            ax[i].imshow(mat, cmap="gist_yarg")
        top_left += F

    start_time = time.perf_counter()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    det_data, obs_data, err_data = dem_sampler.sample(shots=num_shots, return_errors=False, bit_packed=False)
    end_time = time.perf_counter()
    print(f"Stim: noise sampling for {num_shots} shots, elapsed time:", end_time-start_time)


    total_e_hat = np.zeros((num_shots,num_col))
    new_det_data = det_data.copy()
    start_time = time.perf_counter()
    top_left = 0
    i = 0
    osd = False
    while i < num_win:
        mat = chk_submats[i]
        prior = prior_subvecs[i]
        a = anchors[top_left]
        bottom_right = min(top_left + W, len(anchors)-1)
        b = anchors[bottom_right]
        c = anchors[top_left+F] # commit region bottom right

        if i==num_win-1 and osd:
            bpd = BpOsdDecoder(
                mat,
                channel_probs=list(prior),
                max_iter=200,
                bp_method="minimum_sum",
                ms_scaling_factor=1.0,
                osd_method="OSD_CS",
                osd_order=10,
            )
        else:
            bpgdg = bpgdg_decoder(
                mat,
                channel_probs=prior,
                max_iter=max_iter,
                max_iter_per_step=max_iter_per_step,
                max_step=max_step,
                max_tree_depth=max_tree_depth,
                max_side_depth=max_side_depth,
                max_tree_branch_step=max_side_branch_step,
                max_side_branch_step=max_side_branch_step,
                multi_thread=True,
                low_error_mode=low_error_mode,
                gdg_factor=last_win_gdg_factor if (i==num_win-1) else 1.0,
                ms_scaling_factor=last_win_bp_factor if (i==num_win-1) else 1.0,
            )
        num_flag_err = 0
#         if i==num_win - 1: # after gathering hard sample, uncomment these two lines
#             return mat, prior # to get mat and prior for the last window
        detector_win = new_det_data[:,a[0]:b[0]]
        llr_prior = np.log((1.0-prior)/prior)
        sum_wt = 0
        for j in range(num_shots):
            if i==num_win-1 and osd:
                e_hat = bpd.decode(detector_win[j])
                is_flagged = ((mat @ e_hat + detector_win[j]) % 2).any()
            else:
#                 e_hat_osd = bpd.decode(detector_win[j])
                decoding_start_time = time.perf_counter()
                # itt.resume()
                e_hat = bpgdg.decode(detector_win[j])
                # itt.pause()

#                 pm_osd = llr_prior[e_hat_osd.astype(bool)].sum()
#                 pm_gdg = llr_prior[e_hat.astype(bool)].sum()
#                 if pm_osd != pm_gdg:
#                     print(f"osd pm {pm_osd}, gdg pm {pm_gdg}")
                decoding_end_time = time.perf_counter()
                is_flagged = 1 - bpgdg.converge
                if is_flagged: decoding_time.append(decoding_end_time-decoding_start_time)
                
#                 if is_flagged and i==num_win-1:
#                     hard_samples.append(detector_win[j])
            sum_wt += e_hat.sum()                
            num_flag_err += is_flagged
            if i == num_win-1: # last window
                total_e_hat[j][a[1]:b[1]] = e_hat
            else:
                total_e_hat[j][a[1]:c[1]] = e_hat[:c[1]-a[1]]
          
        print(f"Window {i}, average weight {sum_wt/num_shots}")
        print(f"Window {i}, flagged Errors: {num_flag_err}/{num_shots}")

        if i!=num_win - 1:
            new_det_data = (det_data + total_e_hat @ chk.T) % 2
            top_left += F
        else:
            end_time = time.perf_counter()
            print("Elapsed time:", end_time-start_time)    
            print("last round osd", osd)
            flagged_err = ((det_data + total_e_hat @ chk.T) % 2).any(axis=1)
            num_flagged_err = flagged_err.astype(int).sum()
            print(f"Overall Flagged Errors: {num_flagged_err}/{num_shots}")
            logical_err = ((obs_data + total_e_hat @ obs.T) % 2).any(axis=1)
            num_err = np.logical_or(flagged_err, logical_err).astype(int).sum()
            print(f"Logical Errors: {num_err}/{num_shots}")
            p_l = num_err / num_shots
            p_l_per_round = 1-(1-p_l) ** (1/num_repeat)
            print("logical error per round:", p_l_per_round)
        
        if i == num_win-1 and osd:
            break
            
        if i == num_win-1 and (not osd):
            i -= 1
            osd = True
            
        i += 1
        
        
sliding_window_decoder(N=144, p=0.005, num_repeat=4, W=3, F=1, num_shots=5000, max_iter=8, method=1, z_basis=True)