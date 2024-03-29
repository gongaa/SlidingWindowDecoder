#ifndef BPGD_HPP
#define BPGD_HPP
extern "C" {
#include "mod2sparse.h"
}
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>

class BPGD {
public:
    int m, n;
    int A, A_sum, C, D;
    int num_active_vn;
    mod2sparse* pcm;
    double* llr_prior;
    double** llr_posterior;
    char* vn_mask;
    char* vn_degree;
    char* cn_mask;
    char* cn_degree;
    char* error;
    char* syndrome;
    char* temp_syndrome;
    int num_iter;
    int low_error_mode;
    double factor;

    int min_sum_log();
    void init();
    int peel();
    int vn_set_value(int vn, char value);
    int reset(mod2sparse* source_pcm, int* copy_cols, double* source_llr_prior, char* source_syndrome);
    void set_masks(char* source_vn_mask, char* source_cn_mask, char* source_cn_degree);
    int select_vn(int depth, int& guess_vn);
    int decimate_vn_reliable(int depth, double fraction);
    void set_thresh(int A, int B, int C, int D) { this->A = A; this->A_sum = B; this->C = C; this->D = D; }
    double get_pm();
    BPGD() {}
    BPGD(int m, int n, int num_iter, int low_error_mode, double factor);
    // Delete copy constructor and copy assignment operator
    // use smart_ptr to avoid double free
    BPGD(const BPGD&) = delete;
    BPGD& operator=(const BPGD&) = delete;
    ~BPGD();
};

class Barrier {
public:
    Barrier(int count) : thread_count(count), count(count), generation(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        int gen = generation;
        if (--count == 0) {
            generation++;
            count = thread_count;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return gen != generation; });
        }
    }

private:
    std::mutex mutex;
    std::condition_variable cv;
    int thread_count;
    int count;
    int generation;
};

class BPGD_tree_thread : public BPGD {
public:
    bool converge;
    int status;
    int id;
    int finished;
    double min_pm;
    int backup_vn;
    int backup_value;
    int current_depth;
    int max_tree_depth;
    int max_step;
    std::vector<char> backup_vn_mask;
    std::vector<char> backup_cn_mask;
    std::vector<char> backup_cn_degree;
    void do_work(mod2sparse* source_pcm, int* copy_cols, double* source_llr_prior, char* source_syndrome, Barrier& barrier,
                 std::mutex& store_mtx, double& pm, std::vector<char>& min_pm_error);
    BPGD_tree_thread(int m, int n, int num_iter, int max_tree_depth, int max_step, int low_error_mode, double factor);
    BPGD_tree_thread() {}
    BPGD_tree_thread(const BPGD_tree_thread&) = delete;
    BPGD_tree_thread& operator=(const BPGD_tree_thread&) = delete;
};

class BPGD_side_thread: public BPGD {
public:
    bool converge;
    int status;
    int id;
    int finished;
    double min_pm;
    int backup_vn;
    int backup_value;
    int current_depth;
    int max_step;
    void do_work(mod2sparse* source_pcm, int* copy_cols, double* source_llr_prior, char* source_syndrome,
                 std::mutex& mtx, std::condition_variable& cv, Barrier& barrier,
                 std::mutex& store_mtx, double& pm, std::vector<char>& min_pm_error);
    BPGD_side_thread(int m, int n, int num_iter, int max_step, int low_error_mode, double factor): BPGD(m, n, num_iter, low_error_mode, factor), max_step(max_step) { set_thresh(0.0, -10.0, 30.0, 3.0); }
    BPGD_side_thread() {}
    BPGD_side_thread(const BPGD_side_thread&) = delete;
    BPGD_side_thread& operator=(const BPGD_side_thread&) = delete;
};

class BPGD_main_thread: public BPGD {
public:
    int max_step;       // main thread max number of steps
    int max_tree_depth; // where tree thread splitting ends 
    int max_tree_step;  // tree threads, how many more steps after last splitting
    int max_side_depth; // where side thread splitting ends
    int max_side_step;  // side threads, how many more steps after last splitting
    int num_tree_threads;
    int num_side_threads;
    double min_pm;
    std::vector<char> min_pm_error; // avoid using dynamic memory allocation, avoid writing destructor, 
    // requires special handling with move (assignment) constructor when used together with thread
    std::vector<std::thread> threads;
    // use smart pointers for class objects, to avoid writing destructor
    std::vector<std::unique_ptr<BPGD_side_thread>> bpgd_side_vec;
    std::vector<std::unique_ptr<BPGD_tree_thread>> bpgd_tree_vec;
    void do_work(mod2sparse* source_pcm, int* copy_cols, double* source_llr_prior, char* source_syndrome);
    BPGD_main_thread(int m, int n, int num_iter, int max_step, int max_tree_depth, int max_side_depth, int max_tree_step, int max_side_step, int low_error_mode, double factor);
    BPGD_main_thread(int m, int n, int low_error_mode) : BPGD_main_thread(m, n, 6, 25, 3, 10, 10, 10, low_error_mode, 1.0) {}
    BPGD_main_thread() {}
    // Delete copy constructor and copy assignment to prevent copying
    BPGD_main_thread(const BPGD_main_thread&) = delete;
    BPGD_main_thread& operator=(const BPGD_main_thread&) = delete;
    // Optionally implement move constructor and move assignment operator
    BPGD_main_thread(BPGD_main_thread&&) noexcept = default;
    BPGD_main_thread& operator=(BPGD_main_thread&&) noexcept = default;

};

void index_sort(double *v, int *cols, int N);
void mod2sparse_mulvec_cpp(mod2sparse *m, char *u, char *v);
void mod2sparse_free_cpp(mod2sparse *m);
double logaddexp(double x, double y);
double log1pexp(double x);

#endif