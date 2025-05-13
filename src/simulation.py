import numpy as np
import time
from ldpc import BpOsdDecoder
import time
from src.utils import rank
from src.codes_q import *
from src import bpgdg_decoder
from src import bp4_osd

def data_qubit_noise_decoding(code, p, num_shots=1000, osd_orders=[10], osd_factor=0.625, skip_gdg=False,
                              max_step=40, max_tree_step=30, max_iter_per_step=6,
                              extra_decoders=[]):                                                     
    print(f"hx shape {code.hx.shape}, hz_perp shape {code.hz_perp.shape}")
#     print(f"girth hx {find_girth(code.hx)}, hz {find_girth(code.hz)}") # a bit slow for the 882 code
    err = np.random.binomial(1, p, (num_shots, code.N)) # [num_shots, N]
    syndrome = (err @ code.hx.T) % 2 # [num_shots, N_half]
    priors = np.ones(code.N) * p
    for dec in extra_decoders:                                                     
        start_time = time.perf_counter()
        num_err = 0
        num_flag_err = 0
        for i in range(num_shots):
            s = syndrome[i]
            e_hat = dec.decode(s)
            e_diff = (e_hat + err[i]) % 2
            logical_err = ((e_diff @ code.hz_perp.T) % 2).any()
            num_err += logical_err
            num_flag_err += 1 - dec.converge
        print("Extra decoder: num flagged error", num_flag_err)
        print(f"Extra decoder: num logical error {num_err}/{num_shots}, LER {num_err/num_shots}")
        end_time = time.perf_counter()
        print("Elapsed time:", end_time-start_time)  

    # OSD
    start_time = time.perf_counter()
    for order in osd_orders:
        osd_num_err = 0
        osd0_num_err = 0
        bpd = BpOsdDecoder(
            code.hx,
            channel_probs=list(priors),
            max_iter=100,
            bp_method="minimum_sum",
            ms_scaling_factor=osd_factor, # usually {0.5, 0.625, 0.8, 1.0} suffice
            osd_method="OSD_CS",
            osd_order=order, # use -1 for BP alone
        )
        for i in range(num_shots):
            s = syndrome[i]
            e_hat_osd = bpd.decode(s) # can extract osd_0 result via bpd.osd0_decoding when using higher order osd
            e_diff = (e_hat_osd + err[i]) % 2
            logical_err = ((e_diff @ code.hz_perp.T) % 2).any()
            osd_num_err += logical_err
            e_diff = (bpd.osd0_decoding + err[i]) % 2
            logical_err = ((e_diff @ code.hz_perp.T) % 2).any()
            osd0_num_err += logical_err
            
        print(f"OSD order 0: num logical error {osd0_num_err}/{num_shots}, LER {osd0_num_err/num_shots}")
        print(f"OSD order {order}: num logical error {osd_num_err}/{num_shots}, LER {osd_num_err/num_shots}")
    
    end_time = time.perf_counter()
    print("Elapsed time:", end_time-start_time)  
    if skip_gdg: return
    
    # GDG
    bpgdg = bpgdg_decoder(
        code.hx,
        channel_probs=priors,
        max_iter_per_step=max_iter_per_step,
        gdg_factor=0.625,
        max_step=max_step, # have to use larger max_step for longer block-length codes
        max_tree_depth=4,
        max_side_depth=20,
        max_tree_branch_step=max_tree_step,
        max_side_branch_step=max_step-20,
        multi_thread=True,
        low_error_mode=True, # always use low error mode
        # don't care about the rest
        max_iter=24,
        ms_scaling_factor=0.625,
        new_n=code.N
    )

    gdg_num_err = 0
    num_flag_err = 0
    start_time = time.perf_counter()
    for i in range(num_shots):
        s = syndrome[i]
        e_hat_gdg = bpgdg.decode(s)  
        num_flag_err += 1 - bpgdg.converge
        e_diff = (e_hat_gdg + err[i]) % 2
        logical_err = ((e_diff @ code.hz_perp.T) % 2).any()
        gdg_num_err += logical_err


    print("GDG: num flagged error", num_flag_err)
    print(f"GDG: num logical error {gdg_num_err}/{num_shots}, LER {gdg_num_err/num_shots}")
    end_time = time.perf_counter()
    print("Elapsed time:", end_time-start_time)  