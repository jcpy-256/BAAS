
#include <cfloat>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/time.h>

#include <string>
#include <algorithm>

#include "spmUtils.hpp"
using namespace std;


void getSimpleThreadPartition(int *begin, int *end, int n)
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int n_per_thread = (n + nthreads - 1) / nthreads;

  *begin = std::min(n_per_thread * tid, n);
  *end = std::min(*begin + n_per_thread, n);
}

void getLoadBalancedPartition(int *begin, int *end, const int *prefixSum, int n)
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int base = prefixSum[0];
  int total_work = prefixSum[n] - base;
  int work_per_thread = (total_work + nthreads - 1) / nthreads;

  *begin = tid == 0 ? 0 : lower_bound(prefixSum, prefixSum + n, work_per_thread * tid + base) - prefixSum;
  *end = tid == nthreads - 1 ? n : lower_bound(prefixSum, prefixSum + n, work_per_thread * (tid + 1) + base) - prefixSum;

  assert(*begin <= *end);
  assert(*begin >= 0 && *begin <= n);
  assert(*end >= 0 && *end <= n);
}

void getInversePerm(int *inversePerm, const int *perm, int n)
{
  assert(inversePerm != nullptr && perm != nullptr);
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
  {
    inversePerm[perm[i]] = i;
  }
}

bool isPerm(const int *perm, int n)
{
  int *temp = new int[n];
  memcpy(temp, perm, sizeof(int) * n);
  sort(temp, temp + n);
  int *last = unique(temp, temp + n);
  if (last != temp + n)
  {
    memcpy(temp, perm, sizeof(int) * n);
    sort(temp, temp + n);

    for (int i = 0; i < n; ++i)
    {
      if (temp[i] == i - 1)
      {
        printf("%d duplicated\n", i - 1);
        // assert(false);
        return false;
      }
      else if (temp[i] != i)
      {
        printf("%d missed\n", i);
        // assert(false);
        return false;
      }
    }
  }
  delete[] temp;
  return true;
}

bool isInversePerm(const int *perm, const int *inversePerm, int len)
{
  for (int i = 0; i < len; ++i)
  {
    if (inversePerm[perm[i]] != i)
      return false;
  }
  return true;
}
