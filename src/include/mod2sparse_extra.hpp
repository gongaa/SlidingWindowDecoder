#ifndef MOD2SPARSE_EXTRA
#define MOD2SPARSE_EXTRA
extern "C" {
#include "mod2sparse.h"
}
char* decimal_to_binary_reverse(int n, int K);

int mod2sparse_rank(mod2sparse *A);

void LU_forward_backward_solve
        (mod2sparse *L,
         mod2sparse *U,
         int *rows,
         int *cols,
         char *z,
         char *x);

int mod2sparse_decomp_osd
        ( mod2sparse *A,	/* Input matrix, M by N */
          int R,		/* Size of sub-matrix to find LU decomposition of */
          mod2sparse *L,	/* Matrix in which L is stored, M by R */
          mod2sparse *U,	/* Matrix in which U is stored, R by N */
          int *rows,		/* Array where row indexes are stored, M long */
          int *cols		/* Array where column indexes are stored, N long */
        );

void mod2sparse_merge_vec(mod2sparse* m1, char *vec, int n, mod2sparse* m2);

mod2sparse* mod2sparse_allocate_cpp (int u, int v);
void mod2sparse_copycols_cpp (mod2sparse* m1, mod2sparse* m2, int* cols);

#endif