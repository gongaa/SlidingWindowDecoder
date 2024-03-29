#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <iostream>
#include <sched.h>
#include <pthread.h>
#include "bpgd.hpp"
#define MAX_PM 10000.0

int BPGD::peel() {
    mod2entry* e;
    int cn, vn;
    bool degree_check;
    while (true) {
        degree_check = true;
        for (cn=0; cn < this->m; cn++) {
            if (cn_mask[cn] == -1) continue; // already cleared, therefore inactivated
            if (cn_degree[cn] >= 2) continue; // cannot decide any neighboring VN of this CN
            if (cn_degree[cn] <= 0) {
                fprintf(stderr, "EXCEPTION: cn %d deg 0, do not agree with cn_mask, should exit\n", cn);
                cn_mask[cn] = -1; // TODO: find the cause of this
                continue;
            }
            // must be degree 1, find the unique neighboring VN
            degree_check = false;
            vn = -1;
            // iterate through VNs checked by this CN
            e = mod2sparse_first_in_row(this->pcm, cn);
            while (!mod2sparse_at_end(e)) {
                if (this->vn_mask[e->col] != -1) { // inactive VN
                    e = mod2sparse_next_in_row(e);
                    continue;
                }
                vn = e->col;
                break;
            }
            if (vn == -1) {
                return -1;
                // fprintf(stderr, "vn=-1, exit\n");
            }
            if (this->vn_set_value(vn, this->cn_mask[cn]) == -1) return -1;
        }
        if (degree_check) return 0;
    }
    return 0;
}

int BPGD::vn_set_value(int vn, char value) {
    if (this->vn_mask[vn] != -1) {
        if (this->vn_mask[vn] == value) return 0;
        else return -1;
    }
    this->num_active_vn -= 1;
    this->vn_mask[vn] = value;
    this->error[vn] = value;
    mod2entry* e;
    int cn, deg;
    // iterate through all the neighboring CNs
    e = mod2sparse_first_in_col(this->pcm, vn);
    while (!mod2sparse_at_end(e)) {
        if (this->cn_mask[e->row] == -1 || this->cn_degree[e->row] == 0) { // inactive CN
            fprintf(stderr, "vn set value EXCEPTION: VN %d has an inactive CN neighbor %d, exit\n", vn, e->row);
            return -1;
        }
        cn = e->row;
        deg = this->cn_degree[cn] - 1;
        if (value) this->cn_mask[cn] = 1 - this->cn_mask[cn];
        this->cn_degree[cn] = deg;
        if (deg == 0) {
            if (this->cn_mask[cn] != 0) return -1;
            this->cn_mask[cn] = -1; 
        }
        e = mod2sparse_next_in_col(e);
    }
    return 0;

}

void BPGD::init() {
    mod2entry* e;
    double llr;
    for (int vn=0; vn < this->n; vn++) {
        if (this->vn_mask[vn] != -1) continue;
        e = mod2sparse_first_in_col(this->pcm, vn);
        llr = this->llr_prior[vn];
        while (!mod2sparse_at_end(e)) {
            e->bit_to_check = llr;
            e = mod2sparse_next_in_col(e);
        }
    }
    return;
}

int BPGD::min_sum_log() {
    mod2entry* e;
    int cn, vn, sgn;
    bool equal;
    double temp = 0.0;
    for (int it=0; it < this->num_iter; it++) {
        for (cn=0; cn < this->m; cn++) {
            if (this->cn_mask[cn] == -1) continue;
            e = mod2sparse_first_in_row(this->pcm, cn);
            temp = 1e308;
            if (this->cn_mask[cn] == 1) sgn = 1;
            else sgn = 0;

            // first pass, find the min abs value of all incoming messages, determine sign
            while (!mod2sparse_at_end(e)) {
                if (this->vn_mask[e->col] != -1) {
                    e = mod2sparse_next_in_row(e);
                    continue;
                }
                e->check_to_bit = temp;
                e->sgn = sgn;

                // clipping
                if (e->bit_to_check > 50.0) e->bit_to_check = 50.0;
                else if (e->bit_to_check < -50.0) e->bit_to_check = -50.0;

                if (abs(e->bit_to_check) < temp) temp = abs(e->bit_to_check);
                if (e->bit_to_check <= 0) sgn = 1 - sgn;
                e = mod2sparse_next_in_row(e);
            }

            // second pass, set min to second min, others to min
            e = mod2sparse_last_in_row(this->pcm, cn);
            temp = 1e308;
            sgn = 0;
            while (!mod2sparse_at_end(e)) {
                if (this->vn_mask[e->col] != -1) {
                    e = mod2sparse_prev_in_row(e);
                    continue;
                }

                if (temp < e->check_to_bit) e->check_to_bit = temp;
                e->sgn += sgn;

                e->check_to_bit *= this->factor * (e->sgn%2==0 ? 1: -1);

                if (abs(e->bit_to_check) < temp) temp = abs(e->bit_to_check);
                if (e->bit_to_check <= 0) sgn = 1 - sgn;

                e = mod2sparse_prev_in_row(e);
            }
        }

        // bit-to-check messages
        for (vn=0; vn < this->n; vn++) {
            if (this->vn_mask[vn] != -1) continue;
            e = mod2sparse_first_in_col(this->pcm, vn);
            temp = this->llr_prior[vn];

            while (!mod2sparse_at_end(e)) {
                if (this->cn_mask[e->row] == -1) {
                    e = mod2sparse_next_in_col(e);
                    continue;
                }

                e->bit_to_check = temp; // sum from the left to itself
                temp += e->check_to_bit;
                e = mod2sparse_next_in_col(e);
            }
            this->llr_posterior[vn][it % 4] = temp;
            if (temp <= 0) this->error[vn] = 1;
            else this->error[vn] = 0;

            e = mod2sparse_last_in_col(this->pcm, vn);
            temp = 0.0;
            while (!mod2sparse_at_end(e)) {
                if (cn_mask[e->row] == -1) {
                    e = mod2sparse_prev_in_col(e);
                    continue;
                }

                e->bit_to_check += temp;
                temp += e->check_to_bit;
                e = mod2sparse_prev_in_col(e);
            }
        }

        // check if converged
        mod2sparse_mulvec(this->pcm, this->error, this->temp_syndrome);

        equal = true;
        for (cn=0; cn < this->m; cn++) {
            if (this->syndrome[cn] != this->temp_syndrome[cn]) {
                equal = false;
                break;
            }
        }
        if (equal) return 1; // need return here for early stop (once converged)
    }
    return 0;
}

int BPGD::reset(mod2sparse* source_pcm, int* copy_cols, double* source_llr_prior, char* source_syndrome) {
    mod2sparse_free(this->pcm);
    this->pcm = mod2sparse_allocate(m, n);
    mod2sparse_copycols(source_pcm, this->pcm, copy_cols);
    int vn, cn;
    char deg = 0;
    mod2entry* e;
    num_active_vn = this->n;
    for (vn=0; vn < this->n; vn++) this->llr_prior[vn] = source_llr_prior[copy_cols[vn]];
    for (vn=0; vn < this->n; vn++) this->vn_mask[vn] = -1; // all active (not decided)
    for (cn=0; cn < this->m; cn++) this->cn_mask[cn] = source_syndrome[cn];
    for (cn=0; cn < this->m; cn++) { // CN degree
        e = mod2sparse_first_in_row(this->pcm, cn);
        if (e->col == -1) { // CN deg 0, TODO: check this is a correct way of detecting degree 0
            // if (this->cn_mask[cn] != 0) return -1; // no possible solution
            this->cn_degree[cn] = 0;
            this->cn_mask[cn] = -1; // inactivate CN
            continue;
        }
        deg = 0;
        while (!mod2sparse_at_end(e)) {
            deg += 1;
            e = mod2sparse_next_in_row(e);
        }
        this->cn_degree[cn] = deg;
    }
    for (vn=0; vn < this->n; vn++) { // VN degree, must guarantee all VNs have deg > 0 !!!
        e = mod2sparse_first_in_col(this->pcm, vn);
        deg = 0;
        while (!mod2sparse_at_end(e)) {
            deg += 1;
            e = mod2sparse_next_in_col(e);
        }
        this->vn_degree[vn] = deg;
    }
    for (vn=0; vn < this->n; vn++) this->error[vn] = 0;
    for (cn=0; cn < this->m; cn++) this->syndrome[cn] = source_syndrome[cn]; 
    if (this->peel() == -1) return -1; // peel in case there is degree 1 CN
    this->init(); // BP initialization 
    return 0;
}

void BPGD::set_masks(char* source_vn_mask, char* source_cn_mask, char* source_cn_degree) {
    int vn, cn;
    for (vn=0; vn < this->n; vn++) vn_mask[vn] = source_vn_mask[vn];
    for (vn=0; vn < this->n; vn++) error[vn] = vn_mask[vn];
    for (cn=0; cn < this->m; cn++) cn_mask[cn] = source_cn_mask[cn];
    for (cn=0; cn < this->m; cn++) cn_degree[cn] = source_cn_degree[cn];
    this->init();
}

double BPGD::get_pm() {
    double pm = 0;
    for (int vn=0; vn < this->n; vn++) 
        if (error[vn])
            pm += this->llr_prior[vn];
    return pm;
}

int BPGD::decimate_vn_reliable(int depth, double fraction=1.0) {
    int vn, cn, abs_sum_largest_vn = -1, abs_sum_largest_vn_sign = 0;
    mod2entry* e;
    double* history;
    double abs_sum_largest = 0.0, history_sum = 0.0;

    for (vn=0; vn < n; vn++) {
        if (vn_mask[vn] != -1) continue;
        // if (vn_degree[vn] <= 2) continue;
        history = llr_posterior[vn];
        // history_sum = history[0] + history[1] + history[2] + history[3];
        history_sum = history[3];
        if (abs(history_sum) > abs_sum_largest) {
            abs_sum_largest = abs(history_sum);
            abs_sum_largest_vn = vn;
            abs_sum_largest_vn_sign = (history_sum > 0) ? 0 : 1;
        }
    }
    // std::cout << "decimate vn " << abs_sum_largest_vn << " to " << abs_sum_largest_vn_sign << " history sum " << abs_sum_largest << std::endl;
    if (vn_set_value(abs_sum_largest_vn, abs_sum_largest_vn_sign) == -1) {
        std::cout << "vn set value -1" << std::endl;
        return -1;
    }
    if (peel() == -1) {
        return -1;
        std::cout << "peeling failed" << std::endl;
    }
    return 0;
}

int BPGD::select_vn(int current_depth, int& guess_vn) {
    bool all_smaller_than_A, all_negative, all_larger_than_C, all_larger_than_D;
    int vn, cn, i, sum_smallest_vn = -1, sum_smallest_all_neg_vn = -1;
    int num_flip;
    mod2entry* e;
    double* history;
    double llr, sum_smallest = MAX_PM, history_sum = 0.0, sum_smallest_all_neg = MAX_PM;

    for (vn=0; vn < n; vn++) {
        if (vn_mask[vn] != -1) continue; // skip inactive vn
        if (vn_degree[vn] <= 2) continue; // skip degree 1 or 2 vn
        num_flip = 0;
        e = mod2sparse_first_in_col(pcm, vn);
        while (!mod2sparse_at_end(e)) {
            cn = e->row;
            if (cn_mask[cn] == -1) {
                e = mod2sparse_next_in_col(e);
                continue;
            }
            if (syndrome[cn] != temp_syndrome[cn]) num_flip += 1;
            e = mod2sparse_next_in_col(e);
        }

        history = llr_posterior[vn];
        all_smaller_than_A = true;
        all_negative = true;
        all_larger_than_C = true;
        all_larger_than_D = true;
        history_sum = 0.0;
        for (i=0; i<4; i++) {
            llr = history[i];
            history_sum += llr;
            if (llr < C) all_larger_than_C = false;
            if (llr < D) all_larger_than_D = false;
            if (llr > A) all_smaller_than_A = false;
            if (llr > 0) all_negative = false;
        }
        // aggressive decimation
        if (!low_error_mode && all_larger_than_C && current_depth < 4) { if (vn_set_value(vn, 0) == -1) return -1; }
        else if (!low_error_mode && num_flip >= 3 && all_larger_than_D) { if (vn_set_value(vn, 0) == -1) return -1; }
        else if (!low_error_mode && all_smaller_than_A && history_sum < A_sum) { if (vn_set_value(vn, 1) == -1) return -1; }
        else {
            if (history_sum < sum_smallest) {
                sum_smallest = history_sum;
                sum_smallest_vn = vn;
            }
            if (all_negative && history_sum < sum_smallest_all_neg) {
                sum_smallest_all_neg = history_sum;
                sum_smallest_all_neg_vn = vn;
            }
        }
    }

    if (peel() == -1) return -1; // aggressive decimation failed

    if (sum_smallest_all_neg_vn != -1) {
        guess_vn = sum_smallest_all_neg_vn;
        return 1; // favor 1
    } else {
        // if (sum_smallest_vn == -1) std::cerr << "sum smallest vn = -1, no active vn left" << std::endl;
        guess_vn = sum_smallest_vn;
        return (sum_smallest > 0) ? 0 : 1;
    }
}

BPGD::BPGD(int m, int n, int num_iter, int low_error_mode, double factor): 
m(m), n(n), num_iter(num_iter), low_error_mode(low_error_mode), factor(factor) {
    pcm = mod2sparse_allocate(m, n);
    llr_prior = new double[n];
    llr_posterior = new double*[n];
    for (int i=0; i<n; i++) llr_posterior[i] = new double[4];
    vn_mask = new char[n];
    vn_degree = new char[n];
    cn_mask = new char[m];
    cn_degree = new char[m];
    error = new char[n];
    syndrome = new char[m];
    temp_syndrome = new char[m];
}

BPGD::~BPGD() {
    // std::cerr << "in BGPD destructor" << std::endl;
    mod2sparse_free(pcm);
    delete[] llr_prior;
    for (int i=0; i<n; i++) delete[] llr_posterior[i];
    delete[] llr_posterior;
    delete[] vn_mask;
    delete[] vn_degree;
    delete[] cn_mask;
    delete[] cn_degree;
    delete[] error;
    delete[] syndrome;
    delete[] temp_syndrome;
}

using namespace std;
void index_sort(double *v, int *cols, int N) {
    vector<double> idx(N);
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    for (int i=0; i < N; i++) cols[i] = idx[i];
}

void mod2sparse_mulvec_cpp(mod2sparse *m, char *u, char *v) {
    mod2sparse_mulvec(m, u, v);
}

void mod2sparse_free_cpp(mod2sparse *m) {
    mod2sparse_free(m);
}

double log1pexp(double x) {
    // Direct calculation for x when it does not lead to overflow
    if (x > -log(std::numeric_limits<double>::epsilon())) {
        return x + std::log1p(std::exp(-x));
    }
    // For large x to avoid overflow
    return std::log1p(std::exp(x));
}

double logaddexp(double x, double y) {
    double const tmp = x - y;
    if (x == y) return x + M_LN2;
    if (tmp > 0)
        return x + log1pexp(-tmp);
    else if (tmp <= 0)
        return y + log1pexp(tmp);
    return tmp;
}


void set_affinity(std::thread& th, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error setting thread affinity: " << rc << std::endl;
    }
}

BPGD_tree_thread::BPGD_tree_thread(int m, int n, int num_iter, int max_tree_depth, int max_step, int low_error_mode, double factor): 
BPGD(m, n, num_iter, low_error_mode, factor), max_tree_depth(max_tree_depth), max_step(max_step),
backup_vn_mask(n), backup_cn_mask(m), backup_cn_degree(m) {
    set_thresh(-3.0, -16.0, 30.0, 3.0);
}

void BPGD_tree_thread::do_work(mod2sparse* source_pcm, int* copy_cols, double* source_llr_prior, 
char* source_syndrome, Barrier& barrier, mutex& store_mtx, double& main_min_pm, vector<char>& min_pm_error) {
    min_pm = MAX_PM;
    int reset_status = reset(source_pcm, copy_cols, source_llr_prior, source_syndrome);
    barrier.wait();
    if (reset_status == -1) return;
    // choose favor or favor from depth 0-3 based on thread id binary representation
    // store unfavor at depth 3
    // after 10 steps, load backup snapshot and do another 10 steps.
    converge = false;
    bool on_side_branch = false;
    bool favor_direction = false;
    bool saved = false;
    A = -3.0; A_sum = -16.0;
    for (current_depth = 0; current_depth < max_step + max_tree_depth + 1; current_depth++) {
        if (current_depth > 0 && !on_side_branch) A_sum = -12.0;
        converge = min_sum_log();
        if (converge) {
            min_pm = get_pm();
            lock_guard<mutex> lock(store_mtx);
            if (min_pm < main_min_pm) {
                main_min_pm = min_pm;
                for (int vn=0; vn < n; vn++) min_pm_error[vn] = error[vn];
            }
            return;
        }
        int guess_vn = -1;
        int favor = select_vn(current_depth, guess_vn);
        if (favor == -1 || guess_vn == -1) break;
        if (current_depth < max_tree_depth) {
            favor_direction = (id >> (max_tree_depth-1-current_depth)) & 1;
            if (favor_direction == 1) {
                on_side_branch = true;
                A = -0.0; A_sum = -10.0; // enter side branch
            }
            favor = favor_direction ? 1-favor : favor; // depends on the last bit
            // {
            //     lock_guard<mutex> lock(store_mtx);
            //     cerr << "multi: tree thread id " << id << " chooses vn " << guess_vn << " value " << favor << endl;
            // }
        }
        else if (current_depth == max_tree_depth) {
            // store snapshot
            for (int vn=0; vn < n; vn++) backup_vn_mask[vn] = vn_mask[vn];
            for (int cn=0; cn < m; cn++) backup_cn_mask[cn] = cn_mask[cn];
            for (int cn=0; cn < m; cn++) backup_cn_degree[cn] = cn_degree[cn];
            backup_vn = guess_vn;
            backup_value = 1 - favor;
            saved = true;
        }
        if (vn_set_value(guess_vn, favor) == -1) break; // early end
        if (peel() == -1) break;
        // TODO: re-initialize BP if decide for unfavor
        // if (favor_direction) this->init();
    }
    if (!saved) return; // no snapshot was stored
    // load snapshot
    for (int vn=0; vn < n; vn++) vn_mask[vn] = backup_vn_mask[vn];
    for (int vn=0; vn < n; vn++) error[vn] = vn_mask[vn];
    for (int cn=0; cn < m; cn++) cn_mask[cn] = backup_cn_mask[cn];
    for (int cn=0; cn < m; cn++) cn_degree[cn] = backup_cn_degree[cn];
    this->init(); // don't forget to init
    // {
    //     lock_guard<mutex> lock(store_mtx);
    //     cerr << "multi: tree thread load vn " << backup_vn << " with value " << backup_value << endl;
    // }
    if (vn_set_value(backup_vn, backup_value) == -1) return; 
    if (peel() == -1) return;
    current_depth = max_tree_depth + 1;
    for (int i = 0; i < max_step; i++) {
        converge = min_sum_log();
        if (converge) {
            double temp_pm = get_pm();
            if (temp_pm > min_pm) return;
            min_pm = temp_pm;
            lock_guard<mutex> lock(store_mtx);
            if (min_pm < main_min_pm) {
                main_min_pm = min_pm;
                for (int vn=0; vn < n; vn++) min_pm_error[vn] = error[vn];
            }
            return;
        }
        int guess_vn = -1;
        int favor = select_vn(current_depth, guess_vn);
        if (favor == -1 || guess_vn == -1) return; 
        if (vn_set_value(guess_vn, favor) == -1) return;
        if (peel() == -1) return;
        current_depth++;
    }

}

void BPGD_side_thread::do_work(mod2sparse* source_pcm, int* copy_cols, double* source_llr_prior, char* source_syndrome,
mutex& mtx, condition_variable& cv, Barrier& barrier,
mutex& store_mtx, double& main_min_pm, vector<char>& min_pm_error) {
    status = 0;
    min_pm = MAX_PM;
    int reset_status = reset(source_pcm, copy_cols, source_llr_prior, source_syndrome);
    barrier.wait();
    if (reset_status == -1) return;
    // wait for condition variable to become true
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]{ return status != 0; });
    lock.unlock();
    if (status == -1) return;
    // IMPORTANT: don't forget to initialize error
    for (int vn=0; vn < n; vn++) error[vn] = vn_mask[vn];
    // no need to init, already done in reset
    // {   
    //     lock_guard<mutex> lock(store_mtx);
    //     cerr << "multi: side thread depth " << current_depth << " load vn " << backup_vn << " with value " << backup_value << endl;
    // }
    if (vn_set_value(backup_vn, backup_value) == -1) return;
    if (peel() == -1) return;
    converge = false;
    for (int i=0; i < max_step; i++) {
        converge = min_sum_log();
        if (converge) {
            min_pm = get_pm();
            lock_guard<mutex> lock(store_mtx);
            if (min_pm < main_min_pm) {
                main_min_pm = min_pm;
                for (int vn=0; vn < n; vn++) min_pm_error[vn] = error[vn];
            }
            return; // join
        }
        int guess_vn = -1;
        int favor = select_vn(current_depth, guess_vn);
        if (favor == -1 || guess_vn == -1) return; // something wrong with peeling after aggressive decimation
        if (vn_set_value(guess_vn, favor) == -1) return;
        if (peel() == -1) return;
        current_depth++;
    }


}

BPGD_main_thread::BPGD_main_thread(int m, int n, int num_iter, int max_step, int max_tree_depth, int max_side_depth, int max_tree_step, int max_side_step, int low_error_mode, double factor)
: BPGD(m, n, num_iter, low_error_mode, factor), max_step(max_step),
max_tree_depth(max_tree_depth), max_tree_step(max_tree_step),
max_side_depth(max_side_depth), max_side_step(max_side_step),
num_tree_threads((1<<max_tree_depth) - 1),
num_side_threads(max_side_depth-max_tree_depth),
min_pm_error(n) {

    for (int i=0; i < num_tree_threads; ++i) {
        bpgd_tree_vec.push_back(std::make_unique<BPGD_tree_thread>(m, n, num_iter, max_tree_depth, max_tree_step, low_error_mode, factor));
        bpgd_tree_vec.back()->id = i+1;
    }
    for (int i=0; i < num_side_threads; ++i) {
        bpgd_side_vec.push_back(std::make_unique<BPGD_side_thread>(m, n, num_iter, max_side_step, low_error_mode, factor));
        bpgd_side_vec.back()->id = i+1;
    }
    set_thresh(-3.0, -12.0, 30.0, 3.0);
}

void BPGD_main_thread::do_work(mod2sparse* source_pcm, int* copy_cols, double* source_llr_prior, char* source_syndrome) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(num_tree_threads, &cpuset); // main thread is assigned to core 7
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) cerr << "Error setting thread affinity: " << rc << endl;
    min_pm = MAX_PM;
    mutex mtx, store_mtx;
    vector<condition_variable> cv(num_side_threads);
    Barrier barrier(num_tree_threads+num_side_threads+1);
    for (int i=0; i < num_tree_threads; i++) {
        // tree thread
        threads.emplace_back(&BPGD_tree_thread::do_work, bpgd_tree_vec[i].get(), source_pcm, copy_cols, source_llr_prior,
        source_syndrome, ref(barrier), ref(store_mtx), ref(min_pm), ref(min_pm_error));
        set_affinity(threads.back(), i);
    }

    for (int i=0; i < num_side_threads; i++) {
        // side thread
        threads.emplace_back(&BPGD_side_thread::do_work, bpgd_side_vec[i].get(), source_pcm, copy_cols, source_llr_prior,
        source_syndrome, ref(mtx), ref(cv[i]), ref(barrier), ref(store_mtx), ref(min_pm), ref(min_pm_error));
        set_affinity(threads.back(), i + num_tree_threads + 1);
    }

    int reset_status = reset(source_pcm, copy_cols, source_llr_prior, source_syndrome);
    barrier.wait();
    if (reset_status == -1) {
        cerr << "reset status -1" << endl;
        for (auto& th : threads) th.join(); 
        threads.clear();
        return;
    }
    // start from depth 4
    // save the snapshot and guess_vn and unfavor value and depth to side branch threads
    // run to max depth (default 25)
    bool converge = false;
    int current_depth;
    for (current_depth=0; current_depth < max_step; current_depth++) {
        converge = min_sum_log(); // 6 iterations
        int guess_vn = -1;
        A_sum = (current_depth == 0) ? -16.0 : -12.0;
        int favor = select_vn(current_depth, guess_vn);
        if (converge || favor == -1 || guess_vn == -1) {
            // notify later threads no need to do work
            int j = current_depth - max_tree_depth;
            if (j < 0) j = 0;
            for (; j < num_side_threads; j++) {
                lock_guard<mutex> lock(mtx); // release the mutex once goes out of scope
                bpgd_side_vec[j]->status = -1;
                cv[j].notify_one();
            }
            if (!converge) break;
            double temp_pm = get_pm();
            lock_guard<mutex> lock(store_mtx);
            if (temp_pm < min_pm) {
                min_pm = temp_pm;
                for (int vn=0; vn < n; vn++) min_pm_error[vn] = error[vn];
            }
            break;
        }
        if (current_depth >= max_tree_depth && current_depth < max_side_depth) {
            // save snapshot to other thread.
            // change status, signal them to wake up and do work
            int j = current_depth - max_tree_depth;
            lock_guard<mutex> lock(mtx);
            for (int vn=0; vn < n; vn++) bpgd_side_vec[j]->vn_mask[vn] = vn_mask[vn];
            for (int cn=0; cn < m; cn++) bpgd_side_vec[j]->cn_mask[cn] = cn_mask[cn];
            for (int cn=0; cn < m; cn++) bpgd_side_vec[j]->cn_degree[cn] = cn_degree[cn];
            bpgd_side_vec[j]->backup_vn = guess_vn;
            bpgd_side_vec[j]->backup_value = 1 - favor;
            bpgd_side_vec[j]->current_depth = current_depth + 1;
            bpgd_side_vec[j]->status = 1;
            cv[j].notify_one();
        }
        if (vn_set_value(guess_vn, favor) != -1 && peel() != -1) continue;
        else { // notify later threads no need to do work
            int j = current_depth + 1 - max_tree_depth;
            if (j < 0) j = 0;
            for (; j < num_side_threads; j++) {
                lock_guard<mutex> lock(mtx);
                bpgd_side_vec[j]->status = -1;
                cv[j].notify_one();
            }
            break;
        }
    }
    // cerr << "multi  thread ends at depth " << current_depth << " with pm " << get_pm() << endl;
    if (!converge) {
        lock_guard<mutex> lock(store_mtx);
        if (min_pm > MAX_PM - 1.0) { // other branches also not converged
            for (int vn=0; vn < n; vn++) min_pm_error[vn] = error[vn];
        }
    }
    for (auto& th : threads) 
        if (th.joinable()) 
            th.join(); 
    threads.clear();
}