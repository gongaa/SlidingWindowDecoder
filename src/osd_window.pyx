# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
from scipy.sparse import spmatrix
# from libcpp.vector cimport vector

cdef class osd_window:

    def __cinit__(self, parity_check_matrix, **kwargs):

        channel_probs = kwargs.get("channel_probs")
        pre_max_iter = kwargs.get("pre_max_iter", 8) # BP preprocessing on original window PCM
        post_max_iter = kwargs.get("post_max_iter", 100) # BP postprocessing on shortened window PCM
        ms_scaling_factor = kwargs.get("ms_scaling_factor", 1.0)
        new_n = kwargs.get("new_n", None)
        osd_method = kwargs.get("osd_method", "osd_0")
        osd_order = kwargs.get("osd_order", 0)

        self.MEM_ALLOCATED = False

        if isinstance(parity_check_matrix, np.ndarray) or isinstance(parity_check_matrix, spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(parity_check_matrix)}")

        self.m, self.n = parity_check_matrix.shape

        if channel_probs[0] != None:
            if len(channel_probs) != self.n:
                raise ValueError(f"The length of the channel probability vector must be eqaul to the block length n={self.n}.")

        #BP Settings
        self.pre_max_iter = pre_max_iter
        self.post_max_iter = post_max_iter
        self.ms_scaling_factor = ms_scaling_factor

        #memory allocation
        if isinstance(parity_check_matrix, np.ndarray):
            self.H = numpy2mod2sparse(parity_check_matrix) #parity check matrix in sparse form
        elif isinstance(parity_check_matrix, spmatrix):
            self.H = spmatrix2mod2sparse(parity_check_matrix)

        assert self.n == self.H.n_cols # validate number of bits in mod2sparse format
        assert self.m == self.H.n_rows # validate number of checks in mod2sparse format
        self.current_vn = <char*>calloc(self.n, sizeof(char)) # 0, 1 stands for decided value, -1 indicates undecided
        self.current_cn = <char*>calloc(self.m, sizeof(char)) # 0, 1 stands for active checks, -1 indicates already cleared
        self.cn_degree = <char*>calloc(self.m, sizeof(char)) # degree of each CN of the original PCM
        self.current_cn_degree = <char*>calloc(self.m, sizeof(char)) # current degree of each CN, for faster peeling
        # CN degree should be small, in particular, < 255, therefore use char instead of int
        self.synd = <char*>calloc(self.m, sizeof(char)) # syndrome string
        self.bp_decoding_synd = <char*>calloc(self.m, sizeof(char)) # decoded syndrome string
        self.bp_decoding = <char*>calloc(self.n, sizeof(char)) # BP decoding
        self.channel_llr = <double*>calloc(self.n, sizeof(double)) # channel probs
        self.log_prob_ratios = <double**>calloc(self.n, sizeof(double*)) # log probability ratios history
        self.history_length = 4 # BP posterior LLR history length
        for vn in range(self.n):
            self.log_prob_ratios[vn] = <double*>calloc(self.history_length, sizeof(double))
        self.llr_sum = <double*>calloc(self.n, sizeof(double))
        # things for post-processing
        self.cols = <int*>calloc(self.n, sizeof(int)) # for index sort according to BP posterior LLR
        if new_n is None:
            self.new_n = min(self.n, self.m * 2)
        else:
            self.new_n = min(new_n, self.n)


        # OSD
        self.osd0_decoding = <char*>calloc(self.n,sizeof(char)) #the OSD_0 decoding
        self.osdw_decoding = <char*>calloc(self.n,sizeof(char)) #the osd_w decoding
        if str(osd_method).lower() in ['OSD_0','osd_0','0','osd0']:
            osd_method = 0
            osd_order = 0
        elif str(osd_method).lower() in ['osd_e','1','osde','exhaustive','e']:
            osd_method = 1
            if osd_order > 15:
                print("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not recommended. Use the 'osd_cs' method instead.")
        elif str(osd_method).lower() in ['osd_cs','2','osdcs','combination_sweep','combination_sweep','cs']:
            osd_method = 2
        else:
            raise ValueError(f"ERROR: OSD method '{osd_method}' invalid. Please choose from the following methods: 'OSD_0', 'OSD_E' or 'OSD_CS'.")
        self.osd_order = int(osd_order)
        self.osd_method = int(osd_method)


        self.encoding_input_count = 0
        
        if self.osd_order > -1:
            self.rank = mod2sparse_rank(self.H)
            try:
                assert self.osd_order <= (self.new_n - self.rank)
            except AssertionError:
                self.osd_order = -1
                raise ValueError(f"For this code, the OSD order should be set in the range 0<=osd_oder<={self.new_n - self.rank}.")
            self.orig_cols = <int*>calloc(self.n, sizeof(int))
            self.rows = <int*>calloc(self.m, sizeof(int))
            self.k = self.new_n - self.rank

        if self.osd_order > 0:
            self.y = <char*>calloc(self.n, sizeof(char))
            self.g = <char*>calloc(self.m, sizeof(char))
            self.Htx = <char*>calloc(self.m, sizeof(char))
            self.Ht_cols = <int*>calloc(self.k, sizeof(int)) 

        if osd_order == 0: pass
        elif self.osd_order > 0 and self.osd_method == 1: self.osd_e_setup()
        elif self.osd_order > 0 and self.osd_method == 2: self.osd_cs_setup()
        elif self.osd_order == -1: pass
        else: raise Exception(f"ERROR: OSD method '{osd_method}' invalid")


        self.MEM_ALLOCATED=True

        if channel_probs[0] != None: # convert probability to log-likelihood ratio (LLR)
            for vn in range(self.n): self.channel_llr[vn] = log((1-channel_probs[vn]) / channel_probs[vn])

        cdef char deg = 0
        for cn in range(self.m):
            # must guarantee each CN has deg > 0 in numpy preprocessing
            e = mod2sparse_first_in_row(self.H, cn)
            deg = 0
            while not mod2sparse_at_end(e):
                deg += 1
                e = mod2sparse_next_in_row(e)
            self.cn_degree[cn] = deg

        self.min_pm = 0.0 # minimum BP path metric
        self.bp_iteration = 0

    cdef void osd_e_setup(self):
        self.encoding_input_count = long(2 ** self.osd_order)
        self.osdw_encoding_inputs = <char**>calloc(self.encoding_input_count, sizeof(char*))
        for i in range(self.encoding_input_count):
            self.osdw_encoding_inputs[i] = decimal_to_binary_reverse(i, self.new_n - self.rank)

    cdef void osd_cs_setup(self):
        cdef int kset_size = self.new_n - self.rank
        assert self.osd_order <= kset_size
        self.encoding_input_count = kset_size + self.osd_order * (self.osd_order-1) / 2
        self.osdw_encoding_inputs = <char**>calloc(self.encoding_input_count, sizeof(char*))
        cdef int total_count = 0
        for i in range(kset_size):
            self.osdw_encoding_inputs[total_count] = <char*>calloc(kset_size, sizeof(char))
            self.osdw_encoding_inputs[total_count][i] = 1
            total_count += 1

        for i in range(self.osd_order):
            for j in range(self.osd_order):
                if i < j:
                    self.osdw_encoding_inputs[total_count] = <char*>calloc(kset_size, sizeof(char))
                    self.osdw_encoding_inputs[total_count][i] = 1
                    self.osdw_encoding_inputs[total_count][j] = 1
                    total_count += 1

        # print("rank", self.rank)
        # print("total count osd CS", total_count)
        assert total_count == self.encoding_input_count


    cpdef np.ndarray[np.int_t, ndim=1] decode(self, input_vector):
        cdef int input_length = input_vector.shape[0]
        cdef int vn

        if input_length == self.m:
            self.synd = numpy2char(input_vector, self.synd)
            self.reset()
            self.bp_init()
            if self.bp_decode_llr(self.pre_max_iter): # check if preprocessing converged
                self.converge = True
                for vn in range(self.n): 
                    if self.bp_decoding[vn]: self.min_pm += self.channel_llr[vn]
                return char2numpy(self.bp_decoding, self.n)
            else:
                for vn in range(self.n):
                    history = self.log_prob_ratios[vn]
                    self.llr_sum[vn] = history[0] + history[1] + history[2] + history[3]

                index_sort(self.llr_sum, self.cols, self.n)

                for vn in range(self.new_n, self.n):
                    if self.vn_set_value(self.cols[vn], 0) == -1: # decimation failed due to contradiction
                        print("setting vn failed")
                        return char2numpy(self.bp_decoding, self.n)
                for vn in range(self.new_n, self.n):
                    self.bp_decoding[self.cols[vn]] = 0
                if self.peel() == -1: # peeling failed due to contradiction
                    print("peeling failed")
                    return char2numpy(self.bp_decoding, self.n)
                self.bp_init()
                if self.bp_decode_llr(self.post_max_iter):
                    self.converge = True
                    for vn in range(self.n): 
                        if self.bp_decoding[vn]: self.min_pm += self.channel_llr[vn]
                    return char2numpy(self.bp_decoding, self.n)
                elif self.osd_order > -1:
                    self.osd()
                    return char2numpy(self.osdw_decoding, self.n)
        else:
            raise ValueError(f"The input to the ldpc.bp_decoder.decode must be a syndrome (of length={self.m}). The inputted vector has length={input_length}. Valid formats are `np.ndarray` or `scipy.sparse.spmatrix`.")
        
        return char2numpy(self.bp_decoding, self.n)

    cdef int osd(self):
        cdef int vn, cn
        cdef int history_idx = (self.post_max_iter - 1) % self.history_length

        for vn in range(self.n):
            if self.current_vn[vn] == 1:
                self.llr_sum[vn] = -1000
            elif self.current_vn[vn] == 0:
                self.llr_sum[vn] = 1000
            else: # not yet decided
                history = self.log_prob_ratios[vn]
                # self.llr_sum[vn] = history[history_idx] # not as good as sum
                self.llr_sum[vn] = history[0] + history[1] + history[2] + history[3]

        index_sort(self.llr_sum, self.cols, self.n)

        cdef int i, j
        cdef mod2sparse* L
        cdef mod2sparse* U
        L = mod2sparse_allocate_cpp(self.m, self.rank)
        U = mod2sparse_allocate_cpp(self.rank, self.n)

        for vn in range(self.n): 
            self.orig_cols[vn] = self.cols[vn]

        # find the LU decomposition of the ordered matrix
        mod2sparse_decomp_osd(self.H, self.rank, L, U, self.rows, self.cols)
        # solve the syndrome equation with most probable full-rank submatrix
        LU_forward_backward_solve(L, U, self.rows, self.cols, self.synd, self.osd0_decoding)

        self.min_pm = 0.0
        # calculate pm for osd0_decoding
        for vn in range(self.n):
            if self.osd0_decoding[vn]: self.min_pm += self.channel_llr[vn]
            self.osdw_decoding[vn] = self.osd0_decoding[vn] # in case no higher order solution has a smaller pm than osd0

        if self.osd_order == 0:
            mod2sparse_free_cpp(U)
            mod2sparse_free_cpp(L)
            return 1

        # return the columns outside of the information set to their original ordering (the LU decomp scrambles them)
        cdef int counter=0, in_pivot
        cdef mod2sparse* Ht = mod2sparse_allocate_cpp(self.m, self.k)
        for i in range(self.new_n):
            cn = self.orig_cols[i]
            in_pivot = 0
            for j in range(self.rank):
                if self.cols[j] == cn:
                    in_pivot = 1
                    break
            if in_pivot == 0:
                self.cols[counter+self.rank] = cn
                counter += 1
        for i in range(self.k):
            self.Ht_cols[i] = self.cols[i + self.rank]
        # copy into the ordered, full-rank matrix Ht
        mod2sparse_copycols_cpp(self.H, Ht, self.Ht_cols)

        cdef char* x
        cdef long int l
        cdef double pm = 0.0
        for l in range(self.encoding_input_count):
            x = self.osdw_encoding_inputs[l]
            # subtract syndrome caused by x, get new syndrome for the syndrome equation
            mod2sparse_mulvec_cpp(Ht, x, self.Htx)
            for cn in range(self.m):
                self.g[cn] = self.synd[cn] ^ self.Htx[cn]

            LU_forward_backward_solve(L, U, self.rows, self.cols, self.g, self.y)
            for vn in range(self.k):
                self.y[self.Ht_cols[vn]] = x[vn]
            pm = 0.0
            for vn in range(self.n):
                if self.y[vn]: pm += self.channel_llr[vn]
            if pm < self.min_pm:
                self.min_pm = pm
                for vn in range(self.n):
                    self.osdw_decoding[vn] = self.y[vn]

        mod2sparse_free_cpp(Ht)
        mod2sparse_free_cpp(U)
        mod2sparse_free_cpp(L)
        return 1



    cdef void reset(self):
        self.bp_iteration = 0
        self.min_pm = 0.0
        cdef mod2entry *e
        cdef int cn, vn

        for cn in range(self.m):
            self.current_cn_degree[cn] = self.cn_degree[cn] # restore CN degree for peeling
        for cn in range(self.m):
            self.current_cn[cn] = self.synd[cn] # all CN active, copy of syndrome
        for vn in range(self.n):
            self.current_vn[vn] = -1 # all VN active
        for vn in range(self.n):
            self.bp_decoding[vn] = 0

        return


    cdef int peel(self):
        # use activation info in self.current_vn and self.current_cn
        # to do peeling decoding, until no more VN can be decided
        # i.e., all CNs have degree >= 2
        cdef mod2entry *e
        cdef int cn, vn 
        cdef bint degree_check
        while True:
            degree_check = True
            for cn in range(self.m):
                if self.current_cn[cn] == -1: # already cleared, therefore inactivated
                    continue
                if self.current_cn_degree[cn] >= 2: # cannot decide any neighboring VN of this CN
                    continue
                # must be degree 1, find the unique neighboring VN
                if self.current_cn_degree[cn] != 1: # sanity check. TODO: comment out
                    print("in peel, expect cn", cn, "to have degree 1, but has degree", self.current_cn_degree[cn])
                degree_check = False # still need to check all CNs in next loop
                vn = -1
                # iterate through VNs checked by this CN
                e = mod2sparse_first_in_row(self.H, cn)
                while not mod2sparse_at_end(e):
                    if self.current_vn[e.col] != -1: # inactive VN
                        e = mod2sparse_next_in_row(e)
                        continue
                    vn = e.col # found this unique VN
                    break
                if self.vn_set_value(vn, self.current_cn[cn]) == -1:
                    return -1

            if degree_check: # all CNs have degree >= 2, peeling ends
                return 0
        return 0
    
    cdef int vn_set_value(self, vn, value):
        # peel one VN
        if self.current_vn[vn] != -1:
            print("vn", vn, "already decided with value", self.current_vn[vn], "but is set again with value", value)
            if self.current_vn[vn] == value:
                return 0
            else:
                return -1
        self.current_vn[vn] = value
        self.bp_decoding[vn] = value
        cdef mod2entry* e 
        cdef int cn, deg
        # iterate through all the neighboring CNs
        e = mod2sparse_first_in_col(self.H, vn)
        while not mod2sparse_at_end(e):
            if self.current_cn[e.row] == -1: # inactivate CN
                e = mod2sparse_next_in_col(e)
                continue
            cn = e.row
            deg = self.current_cn_degree[cn] - 1
            if value: # change CN node value based on the VN decision value
                self.current_cn[cn] = 1 - self.current_cn[cn] # 0->1, 1->0
            if deg == 0: # this check is cleared, inactivate this check
                if self.current_cn[cn] != 0: # contradiction
                    return -1
                self.current_cn[cn] = -1 
            self.current_cn_degree[cn] = deg
            e = mod2sparse_next_in_col(e)
        return 0

    cdef void bp_init(self):
        # initialisation
        for vn in range(self.n):
            if self.current_vn[vn] != -1:
                continue
            e = mod2sparse_first_in_col(self.H, vn)
            llr = self.channel_llr[vn]
            while not mod2sparse_at_end(e):
                e.bit_to_check = llr
                e = mod2sparse_next_in_col(e)

    cdef int bp_decode_llr(self, max_iter):

        cdef mod2entry *e
        cdef int cn, vn, iteration, sgn
        cdef bint equal
        cdef double temp, alpha

        self.converge = 0
        for iteration in range(max_iter):
            self.bp_iteration += 1
            #min-sum check to bit messages
            alpha = self.ms_scaling_factor
            for cn in range(self.m):
                if self.current_cn[cn] == -1: # inactivate CN
                    continue
                # iterate through all the activate neighboring VNs 
                e = mod2sparse_first_in_row(self.H, cn)
                temp = 1e308

                if self.current_cn[cn] == 1: sgn = 1 # use current_cn instead of self.synd
                else: sgn = 0
                # first pass, find the min abs value of all incoming messages, determine sign
                while not mod2sparse_at_end(e):
                    if self.current_vn[e.col] != -1:
                        e = mod2sparse_next_in_row(e)
                        continue

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
                    if self.current_vn[e.col] != -1:
                        e = mod2sparse_prev_in_row(e)
                        continue

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
                if self.current_vn[vn] != -1:
                    continue

                e = mod2sparse_first_in_col(self.H, vn)
                temp = self.channel_llr[vn]

                while not mod2sparse_at_end(e):
                    if self.current_cn[e.row] == -1:
                        e = mod2sparse_next_in_col(e)
                        continue

                    e.bit_to_check = temp # sum from the left to itself
                    temp += e.check_to_bit
                    e = mod2sparse_next_in_col(e)

                self.log_prob_ratios[vn][iteration % self.history_length] = temp
                if temp <= 0: self.bp_decoding[vn] = 1
                else: self.bp_decoding[vn] = 0

                e = mod2sparse_last_in_col(self.H, vn)
                temp = 0.0
                while not mod2sparse_at_end(e):
                    if self.current_cn[e.row] == -1:
                        e = mod2sparse_prev_in_col(e)
                        continue

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
                self.converge = 1
                return 1

        return 0

    @property
    def bp_iteration(self):
        return self.bp_iteration
    
    @property
    def converge(self):
        return self.converge

    @property
    def min_pm(self):
        return self.min_pm

    @property
    def bp_decoding(self):
        return char2numpy(self.bp_decoding, self.n)

    @property
    def osdw_decoding(self):      
        return char2numpy(self.osdw_decoding,self.n)

    @property
    def osd0_decoding(self):  
        return char2numpy(self.osd0_decoding,self.n)

    @property
    def log_prob_ratios(self):
        cdef np.ndarray[np.float_t, ndim=2] np_array = np.zeros((self.n, self.history_length))
        for i in range(self.n):
            for j in range(self.history_length):
                np_array[i,j] = self.log_prob_ratios[i][j]
        return np_array

    def __dealloc__(self):
        if self.MEM_ALLOCATED:
            free(self.synd)
            free(self.bp_decoding_synd)
            free(self.bp_decoding)
            free(self.channel_llr)
            for i in range(self.n):
                free(self.log_prob_ratios[i])
            free(self.log_prob_ratios)

            free(self.current_vn)
            free(self.current_cn)
            free(self.cn_degree)
            free(self.current_cn_degree)
            free(self.cols)

            # OSD
            free(self.osd0_decoding)
            free(self.osdw_decoding)
            if self.osd_order>-1:
                free(self.rows)
                free(self.orig_cols)
            if self.osd_order>0:
                free(self.Htx)
                free(self.g)
                free(self.y)
                free(self.Ht_cols)
            if self.encoding_input_count!=0:
                for i in range(self.encoding_input_count):
                    free(self.osdw_encoding_inputs[i])

            mod2sparse_free_cpp(self.H)