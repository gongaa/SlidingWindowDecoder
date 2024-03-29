# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix

cdef class bp_history_decoder:
    def __cinit__(self, parity_check_matrix, **kwargs):
        channel_probs = kwargs.get("channel_probs")
        max_iter = kwargs.get("max_iter", 50) # pre-processing BP iterations
        ms_scaling_factor = kwargs.get("ms_scaling_factor", 1.0)

        self.MEM_ALLOCATED = False
        if isinstance(parity_check_matrix, np.ndarray) or isinstance(parity_check_matrix, spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(parity_check_matrix)}")

        self.m, self.n = parity_check_matrix.shape

        if channel_probs[0] != None:
            if len(channel_probs) != self.n:
                raise ValueError(f"The length of the channel probability vector must be eqaul to the block length n={self.n}.")

        self.max_iter = max_iter
        self.ms_scaling_factor = ms_scaling_factor
        self.bp_iteration = 0
        self.converge = False

        if isinstance(parity_check_matrix, np.ndarray):
            self.H = numpy2mod2sparse(parity_check_matrix) #parity check matrix in sparse form
        elif isinstance(parity_check_matrix, spmatrix):
            self.H = spmatrix2mod2sparse(parity_check_matrix)

        assert self.n == self.H.n_cols # validate number of bits in mod2sparse format
        assert self.m == self.H.n_rows # validate number of checks in mod2sparse format
        self.synd = <char*>calloc(self.m, sizeof(char)) # syndrome string
        self.bp_decoding_synd = <char*>calloc(self.m, sizeof(char)) # decoded syndrome string
        self.bp_decoding = <char*>calloc(self.n, sizeof(char)) # BP decoding
        self.channel_llr = <double*>calloc(self.n, sizeof(double)) # channel probs
        self.log_prob_ratios = <double**>calloc(self.n, sizeof(double*)) # log probability ratios history
        self.history_length = 4 # BP posterior LLR history length
        for vn in range(self.n):
            self.log_prob_ratios[vn] = <double*>calloc(self.history_length, sizeof(double))

        # for path metric (PM) calculation
        if channel_probs[0] != None:
            for vn in range(self.n): self.channel_llr[vn] = log((1-channel_probs[vn]) / channel_probs[vn])

    cdef int bp_decode_llr(self): # messages in log-likelihood ratio (LLR) form. CN update rule: min sum 
        cdef mod2entry *e
        cdef int cn, vn, iteration, sgn
        cdef bint equal
        cdef double temp, alpha

        # initialisation
        for vn in range(self.n):
            e = mod2sparse_first_in_col(self.H, vn)
            llr = self.channel_llr[vn]
            while not mod2sparse_at_end(e):
                e.bit_to_check = llr
                e = mod2sparse_next_in_col(e)

        for iteration in range(self.max_iter):
            self.bp_iteration += 1
            # min-sum check to bit messages
            alpha = self.ms_scaling_factor
            for cn in range(self.m):
                # iterate through all the active neighboring VNs 
                e = mod2sparse_first_in_row(self.H, cn)
                temp = 1e308

                if self.synd[cn] == 1: sgn = 1
                else: sgn = 0
                # first pass, find the min abs value of all incoming messages, determine sign
                while not mod2sparse_at_end(e):

                    e.check_to_bit = temp # store min from the left most to itself (not inclusive)
                    e.sgn = sgn

                    # clipping
                    if e.bit_to_check > 50.0: e.bit_to_check = 50.0
                    elif e.bit_to_check < -50.0: e.bit_to_check = -50.0

                    if abs(e.bit_to_check) < temp:
                        temp = abs(e.bit_to_check)
                    if e.bit_to_check <= 0: sgn = 1 - sgn
                    e = mod2sparse_next_in_row(e)

                # second pass, set min to second min, others to min
                e = mod2sparse_last_in_row(self.H, cn)
                temp = 1e308
                sgn = 0
                while not mod2sparse_at_end(e):
                    if temp < e.check_to_bit:
                        e.check_to_bit = temp
                    e.sgn += sgn

                    e.check_to_bit *= ((-1)**e.sgn) * alpha

                    if abs(e.bit_to_check) < temp: # store the min from the right most to itself
                        temp = abs(e.bit_to_check)
                    if e.bit_to_check <= 0: sgn = 1 - sgn

                    e = mod2sparse_prev_in_row(e)

            # bit-to-check messages
            for vn in range(self.n):

                e = mod2sparse_first_in_col(self.H, vn)
                temp = self.channel_llr[vn]

                while not mod2sparse_at_end(e):

                    e.bit_to_check = temp # sum from the left to itself
                    temp += e.check_to_bit
                    e = mod2sparse_next_in_col(e)

                self.log_prob_ratios[vn][iteration % self.history_length] = temp
                if temp <= 0: self.bp_decoding[vn] = 1
                else: self.bp_decoding[vn] = 0

                e = mod2sparse_last_in_col(self.H, vn)
                temp = 0.0
                while not mod2sparse_at_end(e):
                    e.bit_to_check += temp # plus the sum from the right to itself
                    temp += e.check_to_bit
                    e = mod2sparse_prev_in_col(e)

            # check if converged
            mod2sparse_mulvec_cpp(self.H, self.bp_decoding, self.bp_decoding_synd)

            equal = True
            for cn in range(self.m):
                if self.synd[cn] != self.bp_decoding_synd[cn]:
                    equal = False
                    break
            if equal:
                return 1

        return 0


    def __dealloc__(self):
        
        if self.MEM_ALLOCATED:
            free(self.synd)
            free(self.bp_decoding_synd)
            free(self.bp_decoding)
            free(self.channel_llr)

            for i in range(self.n):
                free(self.log_prob_ratios[i])
            free(self.log_prob_ratios)

            mod2sparse_free_cpp(self.H)

    @property
    def converge(self):
        return self.converge

cdef class bpgdg_decoder(bp_history_decoder):
    def __cinit__(self, parith_check_matrix, **kwargs):
        max_iter_per_step = kwargs.get("max_iter_per_step", 6)
        max_step = kwargs.get("max_step", 25)
        max_tree_depth = kwargs.get("max_tree_depth", 3)
        max_side_depth = kwargs.get("max_side_depth", 10)
        max_tree_branch_step = kwargs.get("max_tree_branch_step", 10)
        max_side_branch_step = kwargs.get("max_side_branch_step", 10)
        gdg_factor = kwargs.get("gdg_factor", 1.0)
        new_n = kwargs.get("new_n", None)
        multi_thread = kwargs.get("multi_thread", False)
        low_error_mode = kwargs.get("low_error_mode", False)

        self.MEM_ALLOCATED = False

        self.max_iter_per_step = max_iter_per_step
        self.max_step = max_step
        self.max_tree_depth = max_tree_depth
        self.max_side_depth = max_side_depth
        self.max_tree_branch_step = max_tree_branch_step
        self.max_side_branch_step = max_side_branch_step
        self.max_guess = (2 ** max_tree_depth - 1) * 2 + max_side_depth - max_tree_depth
        self.used_guess = 0

        # things for post-processing
        self.llr_sum = <double*>calloc(self.n, sizeof(double))
        self.cols = <int*>calloc(self.n, sizeof(int)) # for index sort according to BP posterior LLR
        if new_n is None:
            self.new_n = min(self.n, self.m * 2)
        else:
            self.new_n = min(new_n, self.n)
        self.bpgd_error = <char*>calloc(self.new_n, sizeof(char))
        if multi_thread:
            self.bpgd_main_thread = new BPGD_main_thread(self.m, self.new_n, self.max_iter_per_step, self.max_step, 
                self.max_tree_depth, self.max_side_depth, self.max_tree_branch_step, self.max_side_branch_step, low_error_mode, gdg_factor)
        else:
            # for saved guessing history
            # for VN: 0, 1 stands for decided value, -1 indicates undecided
            # for CN: 0, 1 stands for active checks, -1 indicates already cleared
            self.vn_stack = <char**>calloc(self.max_guess, sizeof(char*))
            for i in range(self.max_guess):
                self.vn_stack[i] = <char*>calloc(self.new_n, sizeof(char))
            # peeling wouldn't affect [alive] VN degree, thus no stack for vn_degree
            self.cn_stack = <char**>calloc(self.max_guess, sizeof(char*))
            for i in range(self.max_guess):
                self.cn_stack[i] = <char*>calloc(self.m, sizeof(char))
            self.cn_degree_stack = <char**>calloc(self.max_guess, sizeof(char*))
            for i in range(self.max_guess):
                self.cn_degree_stack[i] = <char*>calloc(self.m, sizeof(char))
            self.decision_value_stack = <char*>calloc(self.max_guess, sizeof(char))
            self.decision_vn_stack = <int*>calloc(self.max_guess, sizeof(int))
            self.alt_depth_stack = <int*>calloc(self.max_guess, sizeof(int))
            self.bpgd = new BPGD(self.m, self.new_n, self.max_iter_per_step, low_error_mode, gdg_factor)

        self.MEM_ALLOCATED=True

        self.min_pm = 100000.0 # minimum BP path metric
        self.multi_thread = multi_thread
        self.low_error_mode = low_error_mode
        self.min_converge_depth = 100
    
    cpdef np.ndarray[np.int_t, ndim=1] decode(self, input_vector):
        cdef int input_length = input_vector.shape[0]

        if input_length == self.m:
            self.synd = numpy2char(input_vector, self.synd)
            if self.bp_decode_llr():
                self.converge = True
                return char2numpy(self.bp_decoding, self.n)
            if self.multi_thread:
                self.gdg_multi_thread()
            else:
                self.gdg()
        else:
            raise ValueError(f"The input to the ldpc.bp_decoder.decode must be a syndrome (of length={self.m}). The inputted vector has length={input_length}. Valid formats are `np.ndarray` or `scipy.sparse.spmatrix`.")
        
        return char2numpy(self.bp_decoding, self.n)

    cdef void gdg_multi_thread(self):
        cdef int vn
        for vn in range(self.n):
            history = self.log_prob_ratios[vn]
            self.llr_sum[vn] = history[0] + history[1] + history[2] + history[3]

        index_sort(self.llr_sum, self.cols, self.n)

        self.bpgd_main_thread.do_work(self.H, self.cols, self.channel_llr, self.synd)
        self.converge = (self.bpgd_main_thread.min_pm < 9999.0)
        for vn in range(self.new_n):
            self.bp_decoding[self.cols[vn]] = self.bpgd_main_thread.min_pm_error[vn]
        for vn in range(self.new_n, self.n):
            self.bp_decoding[self.cols[vn]] = 0


    cdef void gdg(self):
        # if BP doesn't converge, run GDG post-processing
        cdef int i, j, vn, cn, current_depth = 0, temp_converge
        cdef double pm = 0.0
        self.converge = False
        for vn in range(self.n):
            history = self.log_prob_ratios[vn]
            self.llr_sum[vn] = history[0] + history[1] + history[2] + history[3]

        index_sort(self.llr_sum, self.cols, self.n)

        # print("single thread sort", end=" ")
        # for i in range(100):
        #     print(self.cols[i], end=" ")
        # print()

        for vn in range(self.new_n, self.n):
            self.bp_decoding[self.cols[vn]] = 0

        if self.bpgd.reset(self.H, self.cols, self.channel_llr, self.synd) == -1:
            return
        # guessing part reset
        self.min_pm = 10000.0
        self.used_guess = 0
        self.bp_iteration = 0 
        self.min_converge_depth = self.max_step
        # do GDG
        # phase 1: growth of main branch and side branches
        for current_depth in range(self.max_step):
            temp_converge = self.bpgd.min_sum_log()
            if temp_converge:
                self.converge = True
                self.min_converge_depth = current_depth
                pm = self.bpgd.get_pm()
                for vn in range(self.new_n):
                    self.bpgd_error[vn] = self.bpgd.error[vn]
                self.min_pm = pm
                # print("main branch converge with pm", pm)
                break
            if self.select_vn(side_branch=False, current_depth=current_depth) == -1:
                break
        if not self.converge: # still copy main branch decision
            # print("main branch did not converge")
            for vn in range(self.new_n):
                self.bpgd_error[vn] = self.bpgd.error[vn]
        # print("single thread ends at depth", self.min_converge_depth, "with pm", self.min_pm)
        # phase 2: try all the side branches, and select the one that converges and has min pm
        i = 0
        while i < self.used_guess:
            current_depth = self.alt_depth_stack[i]
            if current_depth > self.min_converge_depth: # not necessary to explore this path, pm won't be smaller heuristically
                i += 1
                continue
            # load stored snapshot, bp reinit
            self.bpgd.set_masks(self.vn_stack[i], self.cn_stack[i], self.cn_degree_stack[i])
            # make decision and peel
            # print("single: side branch", i, "at depth", self.alt_depth_stack[i], "load vn", self.decision_vn_stack[i], "with value", self.decision_value_stack[i]);
            if self.bpgd.vn_set_value(self.decision_vn_stack[i], self.decision_value_stack[i]) == -1:
                i += 1
                continue
            if self.bpgd.peel() == -1:
                i += 1
                continue
            for j in range(self.max_side_branch_step):
                current_depth = self.alt_depth_stack[i] + j
                temp_converge = self.bpgd.min_sum_log()
                if temp_converge:
                    self.converge = True
                    pm = self.bpgd.get_pm()
                    # print("single: side branch", i, "converge with pm", pm)
                    if pm < self.min_pm:
                        if current_depth < self.min_converge_depth:
                            self.min_converge_depth = current_depth
                        for vn in range(self.new_n):
                            self.bpgd_error[vn] = self.bpgd.error[vn]
                        self.min_pm = pm
                    break
                if current_depth > self.min_converge_depth + 2: # heuristic early stop
                    break
                if self.select_vn(side_branch=True, current_depth=current_depth) == -1:
                    break
            i += 1

        for vn in range(self.new_n):
            self.bp_decoding[self.cols[vn]] = self.bpgd_error[vn]

    cdef int select_vn(self, side_branch, current_depth):
        cdef double A = -3.0 if not side_branch else 0.0
        cdef double A_sum = -12.0 if not side_branch else -10.0 
        if current_depth == 0: A_sum = -16.0 # TODO: try if this is critical
        cdef double C = 30.0
        cdef double D = 3.0
        cdef bint all_smaller_than_A, all_negative, all_larger_than_C, all_larger_than_D
        cdef int vn, cn, i, sum_smallest_vn = -1, sum_smallest_all_neg_vn = -1
        cdef int vn_degree
        cdef int num_flip 
        cdef mod2entry* e
        cdef int guess_vn = -1 
        cdef char favor = 1, unfavor = 0
        cdef double* history
        cdef double llr, sum_smallest = 10000, history_sum = 0.0, sum_smallest_all_neg = 10000
        cdef bint guess = True

        for vn in range(self.new_n):
            if self.bpgd.vn_mask[vn] != -1: # skip inactive vn
                continue
            if self.bpgd.vn_degree[vn] <= 2: # skip degree 1 or 2 vn
                continue
            # how many syndrome inconsistency can flipping this vn reduce (#unsatisfied CN neighbor)
            num_flip = 0
            e = mod2sparse_first_in_col(self.bpgd.pcm, vn)
            while not mod2sparse_at_end(e):
                if self.bpgd.cn_mask[e.row] == -1:
                    e = mod2sparse_next_in_col(e)
                    continue
                cn = e.row
                if self.bpgd.syndrome[cn] != self.bpgd.temp_syndrome[cn]:
                    num_flip += 1
                e = mod2sparse_next_in_col(e)    

            history = self.bpgd.llr_posterior[vn]
            all_smaller_than_A = True
            all_negative = True 
            all_larger_than_C = True
            all_larger_than_D = True 
            history_sum = 0.0
            for i in range(self.history_length):
                llr = history[i]
                history_sum += llr
                if llr < C:    all_larger_than_C = False
                if llr < D:    all_larger_than_D = False
                if llr > A:    all_smaller_than_A = False
                if llr > 0.0:  all_negative = False
            if (not self.low_error_mode) and all_larger_than_C and current_depth < 4: # use current_depth instead of num_active_vn
                if self.bpgd.vn_set_value(vn, 0) == -1:
                    return -1
            elif (not self.low_error_mode) and num_flip >= 3 and all_larger_than_D:
                if self.bpgd.vn_set_value(vn, 0) == -1:
                    return -1
            elif (not self.low_error_mode) and (all_smaller_than_A and history_sum < A_sum):
                if self.bpgd.vn_set_value(vn, 1) == -1:
                    return -1
            else:
                if history_sum < sum_smallest:
                    sum_smallest = history_sum
                    sum_smallest_vn = vn
                if all_negative and history_sum < sum_smallest_all_neg:
                    sum_smallest_all_neg = history_sum
                    sum_smallest_all_neg_vn = vn
            
        if self.bpgd.peel() == -1: # aggressive decimation failed
            return -1

        if sum_smallest_all_neg_vn != -1:
            guess_vn = sum_smallest_all_neg_vn
            favor = 1
        else:
            guess_vn = sum_smallest_vn
            favor = 0 if sum_smallest > 0 else 1

        unfavor = 1 - favor

        if current_depth > self.min_converge_depth: # side branch
            guess = False
        if (not side_branch) and current_depth >= self.max_side_depth:
            guess = False
        if side_branch and current_depth > self.max_tree_depth: 
            guess = False
        if guess and (self.used_guess < self.max_guess):
            used_guess = self.used_guess
            self.decision_value_stack[used_guess] = unfavor
            self.decision_vn_stack[used_guess] = guess_vn
            self.alt_depth_stack[used_guess] = current_depth + 1
            for vn in range(self.new_n):
                self.vn_stack[used_guess][vn] = self.bpgd.vn_mask[vn]
            for cn in range(self.m):
                self.cn_stack[used_guess][cn] = self.bpgd.cn_mask[cn]
            for cn in range(self.m):
                self.cn_degree_stack[used_guess][cn] = self.bpgd.cn_degree[cn]
            self.used_guess = used_guess + 1

        # print("decide on guess vn", guess_vn, "original", self.cols[guess_vn], "degree", self.bpgd.vn_degree[guess_vn], "value", favor)
        # history = self.bpgd.llr_posterior[guess_vn]
        # print("history", history[0], history[1], history[2], history[3])
        if self.bpgd.vn_set_value(guess_vn, favor) == -1:
            return -1
        if self.bpgd.peel() == -1:
            return -1
        return 0

    def __dealloc__(self):
        
        if self.MEM_ALLOCATED:
            free(self.bpgd_error)
            free(self.llr_sum)
            free(self.cols)

            if self.multi_thread:
                del self.bpgd_main_thread
            else:            
                for i in range(self.max_guess):
                    free(self.vn_stack[i])
                free(self.vn_stack)

                for i in range(self.max_guess):
                    free(self.cn_stack[i])
                free(self.cn_stack)

                for i in range(self.max_guess):
                    free(self.cn_degree_stack[i])
                free(self.cn_degree_stack)

                free(self.decision_value_stack)
                free(self.decision_vn_stack)
                free(self.alt_depth_stack)
                del self.bpgd



cdef class bpgd_decoder(bp_history_decoder):
    def __cinit__(self, parity_check_matrix, **kwargs):
        max_iter_per_step = kwargs.get("max_iter_per_step", 6)
        max_step = kwargs.get("max_step", 25)
        gd_factor = kwargs.get("gd_factor", 1.0)
        new_n = kwargs.get("new_n", None)

        # print("BPGD R =", max_step, "T =", max_iter_per_step)
        self.MEM_ALLOCATED = False

        self.max_iter_per_step = max_iter_per_step
        self.max_step = max_step
        self.min_pm = 10000.0

        self.llr_sum = <double*>calloc(self.n, sizeof(double))
        # things for post-processing
        self.cols = <int*>calloc(self.n, sizeof(int)) # for index sort according to BP posterior LLR
        if new_n is None:
            self.new_n = min(self.n, self.m * 2)
        else:
            self.new_n = min(new_n, self.n)
        self.bpgd_error = <char*>calloc(self.new_n, sizeof(char))
        self.bpgd = new BPGD(self.m, self.new_n, self.max_iter_per_step, False, gd_factor)

        self.MEM_ALLOCATED=True

        self.min_converge_depth = 100
    
    cpdef np.ndarray[np.int_t, ndim=1] decode(self, input_vector):
        cdef int input_length = input_vector.shape[0]

        if input_length == self.m:
            self.synd = numpy2char(input_vector, self.synd)
            if self.max_iter > -1:
                if self.bp_decode_llr():
                    self.converge = True
                    return char2numpy(self.bp_decoding, self.n)
                self.gd()
        else:
            raise ValueError(f"The input to the ldpc.bp_decoder.decode must be a syndrome (of length={self.m}). The inputted vector has length={input_length}. Valid formats are `np.ndarray` or `scipy.sparse.spmatrix`.")
        
        return char2numpy(self.bp_decoding, self.n)


    cdef void gd(self):
        # if BP doesn't converge, run GDG post-processing
        cdef int i, j, vn, cn, current_depth = 0, temp_converge
        cdef double pm = 0.0
        self.converge = False
        for vn in range(self.n):
            history = self.log_prob_ratios[vn]
            self.llr_sum[vn] = history[0] + history[1] + history[2] + history[3]

        index_sort(self.llr_sum, self.cols, self.n)

        for vn in range(self.new_n, self.n):
            self.bp_decoding[self.cols[vn]] = 0

        if self.bpgd.reset(self.H, self.cols, self.channel_llr, self.synd) == -1:
            print("bpgd reset fail")
            return
        # GD reset
        self.min_pm = 10000.0
        self.bp_iteration = 0 
        self.min_converge_depth = self.max_step
        # do GD
        # print("GD begins")
        for current_depth in range(self.max_step):
            temp_converge = self.bpgd.min_sum_log()
            # print("rest vn", self.bpgd.num_active_vn)
            if temp_converge:
                self.converge = True
                self.min_converge_depth = current_depth
                pm = self.bpgd.get_pm()
                for vn in range(self.new_n):
                    self.bpgd_error[vn] = self.bpgd.error[vn]
                self.min_pm = pm
                # print("converge with pm", pm)
                break
            if self.bpgd.decimate_vn_reliable(current_depth, 1.0) == -1:
                break
        if not self.converge: # still copy decision
            # print("did not converge")
            for vn in range(self.new_n):
                self.bpgd_error[vn] = self.bpgd.error[vn]
        # print("min converge depth", self.min_converge_depth)
        for vn in range(self.new_n):
            self.bp_decoding[self.cols[vn]] = self.bpgd_error[vn]


    def __dealloc__(self):
        
        if self.MEM_ALLOCATED:
            free(self.bpgd_error)
            free(self.llr_sum)
            free(self.cols)

            del self.bpgd
