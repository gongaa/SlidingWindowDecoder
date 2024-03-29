#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, calloc, free
from libc.math cimport log, tanh, isnan, abs

from .mod2sparse cimport *
from .c_util cimport numpy2char, char2numpy, stackchar2numpy, numpy2double, double2numpy, spmatrix2char, numpy2int

cdef extern from "bpgd.hpp":
    cdef void index_sort(double *soft_decisions, int *cols, int N)
    cdef void mod2sparse_mulvec_cpp(mod2sparse *m, char *u, char *v)
    cdef void mod2sparse_free_cpp(mod2sparse *m)
    cdef double log1pexp(double x)
    cdef double logaddexp(double x, double y)

cdef extern from "mod2sparse_extra.hpp":
    cdef void mod2sparse_print_terminal (mod2sparse *A)
    cdef int mod2sparse_rank(mod2sparse *A)
    cdef mod2sparse* mod2sparse_allocate_cpp (int u, int v)
    cdef void mod2sparse_copycols_cpp (mod2sparse* m1, mod2sparse* m2, int* cols)
    cdef char* decimal_to_binary_reverse(int n, int K)
    
    cdef void LU_forward_backward_solve(
        mod2sparse *L,
        mod2sparse *U,
        int *rows,
        int *cols,
        char *z,
        char *x)

    cdef int mod2sparse_decomp_osd(
        mod2sparse *A,
        int R,
        mod2sparse *L,
        mod2sparse *U,
        int *rows,
        int *cols)

cdef class bp4_osd:
    cdef mod2sparse* Hx
    cdef mod2sparse* Hz
    cdef int mx, mz, n
    cdef char* synd_x
    cdef char* synd_z
    cdef char* bp_decoding_synd_x
    cdef char* bp_decoding_synd_z
    cdef char* bp_decoding_x
    cdef char* bp_decoding_z
    cdef double* channel_llr_x
    cdef double* channel_llr_y
    cdef double* channel_llr_z
    cdef double* log_prob_ratios_x
    cdef double* log_prob_ratios_y
    cdef double* log_prob_ratios_z
    cdef double* prior_llr_x 
    cdef double* prior_llr_z 

    cdef char* current_vn
    cdef char* current_cn_x
    cdef char* current_cn_z
    cdef int* cols
    cdef double* llr_post
    cdef int bp_iteration, max_iter

    cdef double min_pm
    cdef int converge

    cdef double ms_scaling_factor
    cdef int MEM_ALLOCATED

    # OSD
    cdef char* osd0_decoding_x
    cdef char* osd0_decoding_z
    cdef char* osdw_decoding_x
    cdef char* osdw_decoding_z
    cdef char** osdw_encoding_inputs_x
    cdef char** osdw_encoding_inputs_z
    cdef long int encoding_input_count_x
    cdef long int encoding_input_count_z
    cdef int osd_order
    cdef int osd_method
    cdef int rank_x, rank_z
    cdef int kx, kz
    cdef int* rows_x
    cdef int* rows_z
    cdef int* orig_cols
    cdef int* Ht_cols_x
    cdef int* Ht_cols_z
    cdef char* y
    cdef char* gx
    cdef char* gz
    cdef char* Htx
    cdef char* Htz
    cdef void osd_e_setup(self)
    cdef void osd_cs_setup_x(self)
    cdef void osd_cs_setup_z(self)
    cdef int osd(self, basis)

    cpdef np.ndarray[np.int_t, ndim=2] decode(self, input_vector_x, input_vector_z)
    cpdef np.ndarray[np.int_t, ndim=2] camel_decode(self, input_vector_x, input_vector_z)
    cdef void reset(self)
    cdef double cal_pm(self)
    cdef int vn_set_value(self, vn, value)
    cdef int bp4_decode_llr(self)
    cdef void bp_init(self)
    cdef int vn_update(self, vn)
    cdef int cn_update_all(self, basis)


