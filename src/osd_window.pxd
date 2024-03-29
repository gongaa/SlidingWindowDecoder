#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, calloc, free
from libc.math cimport log, tanh, isnan, abs

from .mod2sparse cimport *
from .c_util cimport numpy2char, char2numpy, numpy2double, double2numpy, spmatrix2char, numpy2int

cdef extern from "bpgd.hpp":
    cdef void index_sort(double *soft_decisions, int *cols, int N)
    cdef void mod2sparse_mulvec_cpp(mod2sparse *m, char *u, char *v)
    cdef void mod2sparse_free_cpp(mod2sparse *m)

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

cdef class osd_window:
    cdef mod2sparse* H
    cdef int m, n, new_n
    cdef char* synd
    cdef char* bp_decoding_synd
    cdef char* bp_decoding
    cdef double* channel_llr


    cdef double** log_prob_ratios
    cdef char* current_vn
    cdef char* current_cn
    cdef char* cn_degree
    cdef char* current_cn_degree
    cdef int* cols
    cdef double* llr_sum
    cdef int history_length, bp_iteration

    cdef double min_pm
    cdef int converge

    cdef int pre_max_iter, post_max_iter 

    cdef double ms_scaling_factor
    cdef int MEM_ALLOCATED

    # OSD
    cdef char* osd0_decoding
    cdef char* osdw_decoding
    cdef char** osdw_encoding_inputs
    cdef long int encoding_input_count
    cdef int osd_order
    cdef int osd_method
    cdef int rank
    cdef int k
    cdef int* rows
    cdef int* orig_cols
    cdef int* Ht_cols
    cdef char* y
    cdef char* g
    cdef char* Htx
    cdef void osd_e_setup(self)
    cdef void osd_cs_setup(self)
    cdef int osd(self)

    cpdef np.ndarray[np.int_t, ndim=1] decode(self, input_vector)
    cdef void reset(self)
    cdef int peel(self)
    cdef int vn_set_value(self, vn, value)
    cdef int bp_decode_llr(self, max_iter)
    cdef void bp_init(self)


