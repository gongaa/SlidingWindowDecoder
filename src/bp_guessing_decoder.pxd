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

    cdef cppclass BPGD:
        int n, m
        int num_active_vn
        mod2sparse* pcm
        double* llr_prior
        double** llr_posterior
        char* vn_mask
        char* vn_degree
        char* cn_mask
        char* cn_degree
        char* error
        char* syndrome
        char* temp_syndrome
        int num_iter
        double factor

        int min_sum_log()
        void init()
        int peel()
        int vn_set_value(int vn, char value)
        int decimate_vn_reliable(int depth, double fraction)
        int reset(mod2sparse* source_pcm, int* copy_cols, double* source_llr_prior, char* source_syndrome)
        void set_masks(char* source_vn_mask, char* source_cn_mask, char* source_cn_degree)
        double get_pm()
        BPGD(int m, int n, int num_iter, int low_error_mode, double factor) except +
        BPGD()

    cdef cppclass BPGD_main_thread(BPGD):
        char* min_pm_error
        double min_pm
        void do_work(mod2sparse* source_pcm, int* copy_cols, double* source_llr_prior, char* source_syndrome);
        BPGD_main_thread(int m, int n, int low_error_mode) except +
        BPGD_main_thread(int m, int n, int num_iter, int max_step, int max_tree_depth, int max_side_depth, int max_tree_step, int max_side_step, int low_error_mode, double factor) except +
        BPGD_main_thread()

cdef class bp_history_decoder:
    cdef mod2sparse* H
    cdef int m, n
    cdef char* synd
    cdef char* bp_decoding_synd
    cdef char* bp_decoding
    cdef double* channel_llr
    cdef double** log_prob_ratios
    cdef int history_length, bp_iteration
    cdef int converge
    cdef int max_iter
    cdef double ms_scaling_factor
    cdef int MEM_ALLOCATED

    cdef int bp_decode_llr(self)

cdef class bpgdg_decoder(bp_history_decoder):
    cdef BPGD* bpgd
    cdef BPGD_main_thread* bpgd_main_thread

    cdef double* llr_sum
    cdef int* cols
    cdef int new_n
    cdef char* bpgd_error
    cdef char** vn_stack
    cdef char** cn_stack
    cdef char** cn_degree_stack
    cdef char* decision_value_stack
    cdef int* decision_vn_stack
    cdef int* alt_depth_stack
    cdef double min_pm
    cdef bint multi_thread
    cdef bint low_error_mode
    cdef bint always_restart
    cdef int min_converge_depth

    cdef int max_step, max_iter_per_step
    cdef int max_tree_depth, max_tree_branch_step
    cdef int max_side_depth, max_side_branch_step
    cdef int max_guess, used_guess

    cpdef np.ndarray[np.int_t, ndim=1] decode(self, input_vector)
    cdef void gdg(self)
    cdef void gdg_multi_thread(self)
    cdef int select_vn(self, side_branch, current_depth)

cdef class bpgd_decoder(bp_history_decoder):
    cdef BPGD* bpgd
    cdef int max_step, max_iter_per_step
    cdef double* llr_sum
    cdef int* cols
    cdef int new_n
    cdef char* bpgd_error
    cdef int min_converge_depth
    cdef double min_pm

    cpdef np.ndarray[np.int_t, ndim=1] decode(self, input_vector)
    cdef void gd(self)