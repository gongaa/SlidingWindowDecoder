# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
from scipy.sparse import spmatrix
# from libcpp.vector cimport vector

cdef class bp4_osd:

    def __cinit__(self, Hx, Hz, **kwargs):

        channel_probs_x = kwargs.get("channel_probs_x")
        channel_probs_y = kwargs.get("channel_probs_y")
        channel_probs_z = kwargs.get("channel_probs_z")
        max_iter = kwargs.get("max_iter", 32) 
        ms_scaling_factor = kwargs.get("ms_scaling_factor", 1.0)
        osd_method = kwargs.get("osd_method", "osd_0")
        osd_order = kwargs.get("osd_order", 0)

        self.MEM_ALLOCATED = False

        if isinstance(Hx, np.ndarray) or isinstance(Hx, spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object.")

        if Hx.shape[1] != Hz.shape[1]:
            raise ValueError(f"Hx, Hz blocklength does not match!")

        self.mx, self.n = Hx.shape
        self.mz = Hz.shape[0]
        self.max_iter = max_iter

        if channel_probs_x[0] != None:
            if len(channel_probs_x) != self.n:
                raise ValueError(f"The length of the channel probability vector must be eqaul to the block length n={self.n}.")

        self.ms_scaling_factor = ms_scaling_factor

        # memory allocation
        if isinstance(Hx, np.ndarray):
            self.Hx = numpy2mod2sparse(Hx) # parity check matrix in sparse form
            self.Hz = numpy2mod2sparse(Hz)
        elif isinstance(Hx, spmatrix):
            self.Hx = spmatrix2mod2sparse(Hx)
            self.Hz = spmatrix2mod2sparse(Hz)

        self.current_vn = <char*>calloc(self.n, sizeof(char))  # 0, 1 stands for decided value, -1 indicates undecided
        self.current_cn_x = <char*>calloc(self.mx, sizeof(char)) # -1 for resolved, 0,1 for current value
        self.current_cn_z = <char*>calloc(self.mz, sizeof(char)) 
        # CN degree should be small, in particular, < 255, therefore use char instead of int
        self.synd_x = <char*>calloc(self.mx, sizeof(char)) # syndrome string on Hx
        self.synd_z = <char*>calloc(self.mz, sizeof(char)) # syndrome string on Hz
        self.bp_decoding_synd_x = <char*>calloc(self.mx, sizeof(char)) # decoded syndrome string for Hx
        self.bp_decoding_synd_z = <char*>calloc(self.mz, sizeof(char)) # decoded syndrome string for Hz
        self.bp_decoding_x = <char*>calloc(self.n, sizeof(char)) # BP decoding, X strings, multiply by Hz
        self.bp_decoding_z = <char*>calloc(self.n, sizeof(char)) # BP decoding, Z strings, multiply by Hx
        self.channel_llr_x = <double*>calloc(self.n, sizeof(double)) # channel probs
        self.channel_llr_y = <double*>calloc(self.n, sizeof(double))
        self.channel_llr_z = <double*>calloc(self.n, sizeof(double))
        self.llr_post = <double*>calloc(self.n, sizeof(double)) # used by OSD to rank columns
        self.log_prob_ratios_x = <double*>calloc(self.n, sizeof(double)) # posterior LLR
        self.log_prob_ratios_y = <double*>calloc(self.n, sizeof(double))
        self.log_prob_ratios_z = <double*>calloc(self.n, sizeof(double))
        self.prior_llr_x = <double*>calloc(self.n, sizeof(double))
        self.prior_llr_z = <double*>calloc(self.n, sizeof(double))

        # things for post-processing
        self.cols = <int*>calloc(self.n, sizeof(int)) # for index sort according to BP posterior LLR

        # OSD
        self.osd0_decoding_x = <char*>calloc(self.n, sizeof(char)) #the OSD_0 decoding
        self.osd0_decoding_z = <char*>calloc(self.n, sizeof(char)) #the OSD_0 decoding
        self.osdw_decoding_x = <char*>calloc(self.n, sizeof(char)) #the osd_w decoding
        self.osdw_decoding_z = <char*>calloc(self.n, sizeof(char)) #the osd_w decoding
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


        self.encoding_input_count_x = 0
        self.encoding_input_count_z = 0
        
        if self.osd_order > -1:
            self.rank_x = mod2sparse_rank(self.Hx)
            self.rank_z = mod2sparse_rank(self.Hz)
            try:
                assert self.osd_order <= min(self.n-self.rank_x, self.n-self.rank_z)
            except AssertionError:
                self.osd_order = -1
                raise ValueError(f"For this code, the OSD order should be set in the range 0<=osd_oder<={self.n - self.rank}.")
            self.orig_cols = <int*>calloc(self.n, sizeof(int))
            self.rows_x = <int*>calloc(self.mx, sizeof(int))
            self.rows_z = <int*>calloc(self.mz, sizeof(int))
            self.kx = self.n - self.rank_x
            self.kz = self.n - self.rank_x

        if self.osd_order > 0:
            self.y = <char*>calloc(self.n, sizeof(char))
            self.gx = <char*>calloc(self.mx, sizeof(char))
            self.gz = <char*>calloc(self.mz, sizeof(char))
            self.Htx = <char*>calloc(self.mx, sizeof(char))
            self.Htz = <char*>calloc(self.mz, sizeof(char))
            self.Ht_cols_x = <int*>calloc(self.kx, sizeof(int)) 
            self.Ht_cols_z = <int*>calloc(self.kz, sizeof(int)) 

        if osd_order == 0: pass
        elif self.osd_order > 0 and self.osd_method == 1: 
            self.osd_e_setup()
        elif self.osd_order > 0 and self.osd_method == 2: 
            self.osd_cs_setup_x()
            self.osd_cs_setup_z()
        elif self.osd_order == -1: pass
        else: raise Exception(f"ERROR: OSD method '{osd_method}' invalid")


        self.MEM_ALLOCATED=True

        if channel_probs_x[0] != None: # convert probability to log-likelihood ratio (LLR)
            for vn in range(self.n): 
                num = channel_probs_x[vn] + channel_probs_y[vn] + channel_probs_z[vn]
                num = 1.0 - num
                self.channel_llr_x[vn] = log(num / channel_probs_x[vn])
                self.channel_llr_y[vn] = log(num / channel_probs_y[vn])
                self.channel_llr_z[vn] = log(num / channel_probs_z[vn])
                denom = channel_probs_x[vn] + channel_probs_y[vn]
                self.prior_llr_x[vn] = log((1.0-denom) / denom) # for Hx
                denom = channel_probs_z[vn] + channel_probs_y[vn]
                self.prior_llr_z[vn] = log((1.0-denom) / denom) # for Hz

        self.min_pm = 0.0 # minimum BP path metric
        self.bp_iteration = 0

    cdef void osd_e_setup(self):
        encoding_input_count = long(2 ** self.osd_order)
        self.encoding_input_count_x = encoding_input_count
        self.encoding_input_count_z = encoding_input_count
        self.osdw_encoding_inputs_x = <char**>calloc(encoding_input_count, sizeof(char*))
        self.osdw_encoding_inputs_z = <char**>calloc(encoding_input_count, sizeof(char*))
        for i in range(encoding_input_count):
            self.osdw_encoding_inputs_x[i] = decimal_to_binary_reverse(i, self.n - self.rank_x)
        for i in range(encoding_input_count):
            self.osdw_encoding_inputs_z[i] = decimal_to_binary_reverse(i, self.n - self.rank_z)

    cdef void osd_cs_setup_x(self):

        cdef int kset_size = self.n - self.rank_x
        assert self.osd_order <= kset_size
        self.encoding_input_count_x = kset_size + self.osd_order * (self.osd_order-1) / 2
        self.osdw_encoding_inputs_x = <char**>calloc(self.encoding_input_count_x, sizeof(char*))
        cdef int total_count = 0
        for i in range(kset_size):
            self.osdw_encoding_inputs_x[total_count] = <char*>calloc(kset_size, sizeof(char))
            self.osdw_encoding_inputs_x[total_count][i] = 1
            total_count += 1

        for i in range(self.osd_order):
            for j in range(self.osd_order):
                if i < j:
                    self.osdw_encoding_inputs_x[total_count] = <char*>calloc(kset_size, sizeof(char))
                    self.osdw_encoding_inputs_x[total_count][i] = 1
                    self.osdw_encoding_inputs_x[total_count][j] = 1
                    total_count += 1

        assert total_count == self.encoding_input_count_x

    cdef void osd_cs_setup_z(self):

        cdef int kset_size = self.n - self.rank_z
        assert self.osd_order <= kset_size
        self.encoding_input_count_z = kset_size + self.osd_order * (self.osd_order-1) / 2
        self.osdw_encoding_inputs_z = <char**>calloc(self.encoding_input_count_z, sizeof(char*))
        cdef int total_count = 0
        for i in range(kset_size):
            self.osdw_encoding_inputs_z[total_count] = <char*>calloc(kset_size, sizeof(char))
            self.osdw_encoding_inputs_z[total_count][i] = 1
            total_count += 1

        for i in range(self.osd_order):
            for j in range(self.osd_order):
                if i < j:
                    self.osdw_encoding_inputs_z[total_count] = <char*>calloc(kset_size, sizeof(char))
                    self.osdw_encoding_inputs_z[total_count][i] = 1
                    self.osdw_encoding_inputs_z[total_count][j] = 1
                    total_count += 1

        assert total_count == self.encoding_input_count_z

    cpdef np.ndarray[np.int_t, ndim=2] decode(self, input_vector_x, input_vector_z):
        cdef int input_length_x = input_vector_x.shape[0]
        cdef int vn

        if input_vector_x.shape[0] == self.mx and input_vector_z.shape[0] == self.mz:
            self.synd_x = numpy2char(input_vector_x, self.synd_x)
            self.synd_z = numpy2char(input_vector_z, self.synd_z)
            self.reset()
            self.bp_init()
            if self.bp4_decode_llr(): # check if preprocessing converged
                self.converge = True
                # self.min_pm = self.cal_pm()
                for vn in range(self.n):
                    self.osd0_decoding_x[vn] = self.bp_decoding_x[vn]
                    self.osd0_decoding_z[vn] = self.bp_decoding_z[vn]
                return stackchar2numpy(self.bp_decoding_x, self.bp_decoding_z, self.n)
            elif self.osd_order > -1:
                self.osd('x')
                self.osd('z')
                # self.min_pm = self.cal_pm()
                return stackchar2numpy(self.osdw_decoding_x, self.osdw_decoding_z, self.n)
        else:
            raise ValueError(f"The input to the bp4_osd.decode must be a syndrome (of length={self.mx}).")
        
        return stackchar2numpy(self.bp_decoding_x, self.bp_decoding_z, self.n)

    cpdef np.ndarray[np.int_t, ndim=2] camel_decode(self, input_vector_x, input_vector_z):
        cdef int input_length_x = input_vector_x.shape[0]
        cdef int vn

        if input_vector_x.shape[0] == self.mx and input_vector_z.shape[0] == self.mz:
            self.synd_x = numpy2char(input_vector_x, self.synd_x)
            self.synd_z = numpy2char(input_vector_z, self.synd_z)
            self.min_pm = 10000.0
            for value in range(4):
                self.reset()
                self.bp_init()
                self.vn_set_value(self.n-1, value)
                if self.bp4_decode_llr(): # check if preprocessing converged
                    self.converge = True
                    pm = self.cal_pm()
                    if pm < self.min_pm:
                        self.min_pm = pm
                        for vn in range(self.n):
                            self.osd0_decoding_x[vn] = self.bp_decoding_x[vn]
                            self.osd0_decoding_z[vn] = self.bp_decoding_z[vn]
            if self.min_pm < 9999.0:
                self.converge = True
            return stackchar2numpy(self.osd0_decoding_x, self.osd0_decoding_z, self.n)
        else:
            raise ValueError(f"The input to the bp4_osd.decode must be a syndrome (of length={self.mx}).")
        return stackchar2numpy(self.osd0_decoding_x, self.osd0_decoding_z, self.n)

    cdef double cal_pm(self):
        pm = 0.0
        for vn in range(self.n):
            if (self.bp_decoding_x[vn] and self.bp_decoding_z[vn]):
                pm += self.channel_llr_y[vn]
            elif self.bp_decoding_x[vn]:
                pm += self.channel_llr_x[vn]
            elif self.bp_decoding_z[vn]:
                pm += self.channel_llr_z[vn]
        return pm

    cdef int osd(self, basis):
        cdef int vn, cn

        if basis == 'x':
            rank = self.rank_x
            m = self.mx
            k = self.kx
            H = self.Hx
            rows = self.rows_x
            synd = self.synd_x
            osd0_decoding = self.osd0_decoding_z
            osdw_decoding = self.osdw_decoding_z
            encoding_input_count = self.encoding_input_count_x
            osdw_encoding_inputs = self.osdw_encoding_inputs_x
            g = self.gx
            Htx = self.Htx
            Ht_cols = self.Ht_cols_x
            channel_llr = self.prior_llr_x
            for vn in range(self.n):
                self.llr_post[vn] = log1pexp(-1.*self.log_prob_ratios_x[vn]) - logaddexp(-1.*self.log_prob_ratios_y[vn], -1.*self.log_prob_ratios_z[vn])
        elif basis == 'z':
            rank = self.rank_z
            m = self.mz
            k = self.kx
            H = self.Hz
            rows = self.rows_z
            synd = self.synd_z
            osd0_decoding = self.osd0_decoding_x
            osdw_decoding = self.osdw_decoding_x
            encoding_input_count = self.encoding_input_count_z
            osdw_encoding_inputs = self.osdw_encoding_inputs_z
            g = self.gz
            Htx = self.Htz
            Ht_cols = self.Ht_cols_z
            channel_llr = self.prior_llr_z
            for vn in range(self.n):
                self.llr_post[vn] = log1pexp(-1.*self.log_prob_ratios_z[vn]) - logaddexp(-1.*self.log_prob_ratios_y[vn], -1.*self.log_prob_ratios_x[vn])

        index_sort(self.llr_post, self.cols, self.n)

        cdef int i, j
        cdef mod2sparse* L
        cdef mod2sparse* U
        L = mod2sparse_allocate_cpp(m, rank)
        U = mod2sparse_allocate_cpp(rank, self.n)

        for vn in range(self.n): 
            self.orig_cols[vn] = self.cols[vn]

        # find the LU decomposition of the ordered matrix
        mod2sparse_decomp_osd(H, rank, L, U, rows, self.cols)
        # solve the syndrome equation with most probable full-rank submatrix
        LU_forward_backward_solve(L, U, rows, self.cols, synd, osd0_decoding)

        min_pm = 0.0
        # calculate pm for osd0_decoding
        for vn in range(self.n):
            if osd0_decoding[vn]: min_pm += channel_llr[vn]
            osdw_decoding[vn] = osd0_decoding[vn] # in case no higher order solution has a smaller pm than osd0

        if self.osd_order == 0:
            mod2sparse_free_cpp(U)
            mod2sparse_free_cpp(L)
            return 1

        # return the columns outside of the information set to their original ordering (the LU decomp scrambles them)
        cdef int counter=0, in_pivot
        cdef mod2sparse* Ht = mod2sparse_allocate_cpp(m, k)
        for i in range(self.n):
            cn = self.orig_cols[i]
            in_pivot = 0
            for j in range(rank):
                if self.cols[j] == cn:
                    in_pivot = 1
                    break
            if in_pivot == 0:
                self.cols[counter+rank] = cn
                counter += 1
        for i in range(k):
            Ht_cols[i] = self.cols[i + rank]
        # copy into the ordered, full-rank matrix Ht
        mod2sparse_copycols_cpp(H, Ht, Ht_cols)

        cdef char* x
        cdef long int l
        cdef double pm = 0.0
        for l in range(encoding_input_count):
            x = osdw_encoding_inputs[l]
            # subtract syndrome caused by x, get new syndrome for the syndrome equation
            mod2sparse_mulvec_cpp(Ht, x, Htx)
            for cn in range(m):
                g[cn] = synd[cn] ^ Htx[cn]

            LU_forward_backward_solve(L, U, rows, self.cols, g, self.y)
            for vn in range(k):
                self.y[Ht_cols[vn]] = x[vn]
            pm = 0.0
            for vn in range(self.n):
                if self.y[vn]: pm += channel_llr[vn]
            if pm < min_pm:
                min_pm = pm
                for vn in range(self.n):
                    osdw_decoding[vn] = self.y[vn]

        mod2sparse_free_cpp(Ht)
        mod2sparse_free_cpp(U)
        mod2sparse_free_cpp(L)
        return 1


    cdef void reset(self):
        self.bp_iteration = 0
        cdef mod2entry *e
        cdef int cn, vn

        for cn in range(self.mx):
            self.current_cn_x[cn] = self.synd_x[cn] # all CN active, copy of syndrome
        for cn in range(self.mz):
            self.current_cn_z[cn] = self.synd_z[cn] # all CN active, copy of syndrome
        for vn in range(self.n):
            self.current_vn[vn] = -1 # all VN active
        for vn in range(self.n):
            self.bp_decoding_x[vn] = 0
            self.bp_decoding_z[vn] = 0

        return


    cdef int vn_set_value(self, vn, value): # TODO: value can be 0 (I), 1 (X), 2 (Z), 3 (Y)
        if self.current_vn[vn] != -1:
            print("vn", vn, "already decided with value", self.current_vn[vn], "but is set again with value", value)
            if self.current_vn[vn] == value:
                return 0
            else:
                return -1
        self.current_vn[vn] = value
        x = value % 2
        z = int(value / 2)
        self.bp_decoding_x[vn] = x
        self.bp_decoding_z[vn] = z
        cdef mod2entry* e 
        cdef int cn, deg
        # iterate through all the neighboring CNs
        e = mod2sparse_first_in_col(self.Hx, vn)
        while not mod2sparse_at_end(e):
            if self.current_cn_x[e.row] == -1: # inactivate CN
                e = mod2sparse_next_in_col(e)
                continue
            cn = e.row
            if z: # change CN node value based on the VN decision value
                self.current_cn_x[cn] = 1 - self.current_cn_x[cn] # 0->1, 1->0
            e = mod2sparse_next_in_col(e)

        e = mod2sparse_first_in_col(self.Hz, vn)
        while not mod2sparse_at_end(e):
            if self.current_cn_z[e.row] == -1: # inactivate CN
                e = mod2sparse_next_in_col(e)
                continue
            cn = e.row
            if x: # change CN node value based on the VN decision value
                self.current_cn_z[cn] = 1 - self.current_cn_z[cn] # 0->1, 1->0
            e = mod2sparse_next_in_col(e)
        return 0

    cdef void bp_init(self): # TODO: init
        # initialisation
        for vn in range(self.n):
            if self.current_vn[vn] != -1:
                continue
            llrx = self.channel_llr_x[vn]
            llry = self.channel_llr_y[vn]
            llrz = self.channel_llr_z[vn]
            msg_x = log1pexp(-1.*llrx) - logaddexp(-1.*llry, -1.*llrz) # to Hx
            e = mod2sparse_first_in_col(self.Hx, vn)
            while not mod2sparse_at_end(e):
                e.bit_to_check = msg_x
                e = mod2sparse_next_in_col(e)
            msg_z = log1pexp(-1.*llrz) - logaddexp(-1.*llry, -1.*llrz) # to Hz
            e = mod2sparse_first_in_col(self.Hz, vn)
            while not mod2sparse_at_end(e):
                e.bit_to_check = msg_z
                e = mod2sparse_next_in_col(e)

    cdef int bp4_decode_llr(self):

        cdef mod2entry *e
        cdef int cn, vn, iteration, sgn
        cdef bint equal
        cdef double temp, alpha

        self.converge = 0
        for iteration in range(self.max_iter):
            self.bp_iteration += 1
            # min-sum check to bit messages
            self.cn_update_all('x')
            self.cn_update_all('z')
            # bit-to-check messages
            for vn in range(self.n):
                if self.current_vn[vn] != -1:
                    continue
                self.vn_update(vn)

            # check if converged
            mod2sparse_mulvec_cpp(self.Hx, self.bp_decoding_z, self.bp_decoding_synd_x)
            mod2sparse_mulvec_cpp(self.Hz, self.bp_decoding_x, self.bp_decoding_synd_z)

            equal = True
            for cn in range(self.mx):
                if self.synd_x[cn] != self.bp_decoding_synd_x[cn]:
                    equal = False
                    break
            if equal:
                for cn in range(self.mz):
                    if self.synd_z[cn] != self.bp_decoding_synd_z[cn]:
                        equal = False
                        break
            if equal:
                self.converge = 1
                return 1

        return 0   

    cdef int cn_update_all(self, basis):
        if basis == 'x':
            m = self.mx
            pcm = self.Hx
            cn_mask = self.current_cn_x
        elif basis == 'z':
            m = self.mz
            pcm = self.Hz
            cn_mask = self.current_cn_z

        for cn in range(m):
            # iterate through all the activate neighboring VNs 
            e = mod2sparse_first_in_row(pcm, cn)
            temp = 1e308

            if cn_mask[cn] == 1: sgn = 1 # use current_cn instead of self.synd
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
            e = mod2sparse_last_in_row(pcm, cn)
            temp = 1e308
            sgn = 0
            while not mod2sparse_at_end(e):
                if temp < e.check_to_bit:
                    e.check_to_bit = temp
                e.sgn += sgn

                e.check_to_bit *= ((-1)**e.sgn) * self.ms_scaling_factor 

                if abs(e.bit_to_check) < temp: # store the min from the right most to itself
                    temp = abs(e.bit_to_check)
                if e.bit_to_check <= 0: sgn = 1 - sgn

                e = mod2sparse_prev_in_row(e)



    cdef int vn_update(self, vn):

        llrx = self.channel_llr_x[vn]
        llry = self.channel_llr_y[vn]
        llrz = self.channel_llr_z[vn]

        cdef double llrx_hx = 0.0
        e = mod2sparse_first_in_col(self.Hz, vn)
        while not mod2sparse_at_end(e):
            llrx_hx += e.check_to_bit
            e = mod2sparse_next_in_col(e)

        cdef double llrz_hz = 0.0
        e = mod2sparse_first_in_col(self.Hx, vn)
        while not mod2sparse_at_end(e):
            llrz_hz += e.check_to_bit
            e = mod2sparse_next_in_col(e)

        llry_all = llrx_hx + llrz_hz + llry
        llrx_hx = llrx_hx + llrx
        llrz_hz = llrz_hz + llrz

        self.log_prob_ratios_x[vn] = llrx_hx
        self.log_prob_ratios_y[vn] = llry_all
        self.log_prob_ratios_z[vn] = llrz_hz
        cdef int idx
        if 0 < llrx_hx and 0 < llry_all and 0 < llrz_hz:
            idx = 0
        elif llrx_hx < llry_all and llrx_hx < llrz_hz:
            idx = 1
        elif llry_all > llrz_hz: # llrz the smallest
            idx = 2
        else: # llry the smallest
            idx = 3

        self.bp_decoding_x[vn] = idx % 2
        self.bp_decoding_z[vn] = int(idx/2)

        num_hx = log1pexp(-1.*llrx_hx)
        e = mod2sparse_first_in_col(self.Hx, vn)
        while not mod2sparse_at_end(e):
            msg_x = e.check_to_bit
            llrz_hx = llrz_hz - msg_x
            llry_hx = llry_all - msg_x
            denom_hx = logaddexp(-1.*llrz_hx, -1.*llry_hx)
            e.bit_to_check = num_hx - denom_hx
            e = mod2sparse_next_in_col(e)

        num_hz = log1pexp(-1.*llrz_hz)
        e = mod2sparse_first_in_col(self.Hz, vn)
        while not mod2sparse_at_end(e):
            msg_z = e.check_to_bit
            llrx_hz = llrx_hx - msg_z
            llry_hz = llry_all - msg_z
            denom_hz = logaddexp(-1.*llrx_hz, -1.*llry_hz)
            e.bit_to_check = num_hz - denom_hz
            e = mod2sparse_next_in_col(e)


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
    def bp_decoding_x(self):
        return char2numpy(self.bp_decoding_x, self.n)

    @property
    def bp_decoding_z(self):
        return char2numpy(self.bp_decoding_z, self.n)

    @property
    def osdw_decoding_x(self):      
        return char2numpy(self.osdw_decoding_x,self.n)

    @property
    def osdw_decoding_z(self):      
        return char2numpy(self.osdw_decoding_z,self.n)

    @property
    def osd0_decoding_x(self):  
        return char2numpy(self.osd0_decoding_x,self.n)

    @property
    def osd0_decoding_z(self):  
        return char2numpy(self.osd0_decoding_z,self.n)

    @property
    def log_prob_ratios(self):
        cdef np.ndarray[np.float_t, ndim=2] np_array = np.zeros((self.n, 3))
        for i in range(self.n):
            np_array[i,0] = self.log_prob_ratios_x[i]
            np_array[i,1] = self.log_prob_ratios_y[i]
            np_array[i,2] = self.log_prob_ratios_z[i]
        return np_array

    def __dealloc__(self):
        if self.MEM_ALLOCATED:
            free(self.synd_x)
            free(self.synd_z)
            free(self.bp_decoding_synd_x)
            free(self.bp_decoding_synd_z)
            free(self.bp_decoding_x)
            free(self.bp_decoding_z)
            free(self.channel_llr_x)
            free(self.channel_llr_y)
            free(self.channel_llr_z)
            free(self.log_prob_ratios_x)
            free(self.log_prob_ratios_y)
            free(self.log_prob_ratios_z)
            free(self.prior_llr_x)
            free(self.prior_llr_z)
            free(self.llr_post)

            free(self.current_vn)
            free(self.current_cn_x)
            free(self.current_cn_z)
            free(self.cols)

            # OSD
            free(self.osd0_decoding_x)
            free(self.osd0_decoding_z)
            free(self.osdw_decoding_x)
            free(self.osdw_decoding_z)
            if self.osd_order>-1:
                free(self.rows_x)
                free(self.rows_z)
                free(self.orig_cols)
            if self.osd_order>0:
                free(self.y)
                free(self.gz)
                free(self.gx)
                free(self.Htx)
                free(self.Htz)
                free(self.Ht_cols_x)
                free(self.Ht_cols_z)
            if self.encoding_input_count_x!=0:
                for i in range(self.encoding_input_count_x):
                    free(self.osdw_encoding_inputs_x[i])
            if self.encoding_input_count_z!=0:
                for i in range(self.encoding_input_count_z):
                    free(self.osdw_encoding_inputs_z[i])

            mod2sparse_free_cpp(self.Hx)
            mod2sparse_free_cpp(self.Hz)