#ifndef MMIO_HIGHLEVEL
#define MMIO_HIGHLEVEL

void exclusive_scan(int *input, int length);

void quicksort(int *idx, double *w, int left, int right);

int mmio_feature(int *m, int *n, int *nnz, int *isSymmetric, char *matrix_type, const char *filename);

int mmio_info(int *m, int *n, int *nnz, int *isSymmetric, const char *filename);

int mmio_data(int *csrRowPtr, int *csrColIdx, double *csrVal, const char *filename);

// read matrix infomation from mtx file buf don't processing symmetric matrix
int mmio_allinone(int *m, int *n, int *nnz, int *isSymmetric,
                  int **cooRowIdx, int **cooColIdx, double **cooVal,
                  const char *filename);

// read matrix infomation from mtx file
int mmio_allinone_coo(int *m, int *n, int *nnz, int *isSymmetric,
                      int **cooRowIdx, int **cooColIdx, double **cooVal,
                      const char *filename);

int mmio_allinone_csr(int *m, int *n, int *nnz, int *isSymmetric,
                      int **csrRowPtr, int **csrColIdx, double **csrVal,
                      const char *filename);

int mmio_allinone_csc(int *m, int *n, int *nnz, int *isSymmetric,
                      int **cscColPtr, int **cscRowIdx, double **cscVal,
                      const char *filename);

#endif