#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>


// len: the length of array
template <typename T>
void prefixSumSingle(T *array, int len)
{
  for (int i = 1; i < len; i++)
  {
    array[i] += array[i - 1];
  }
}

template <typename T>
void prefixSum(T *in_out, T *sum, T *workspace)
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  workspace[tid + 1] = *in_out;

#pragma omp barrier
#pragma omp master
  {
    workspace[0] = 0;
    int i;
    for (i = 1; i < nthreads; i++)
    {
      workspace[i + 1] += workspace[i];
    }
    *sum = workspace[nthreads];
  }
#pragma omp barrier

  *in_out = workspace[tid];
}

template <typename T>
void prefixSumMultiple(T *in_out, T *sum, int n, T *workspace)
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int i;
  for (i = 0; i < n; i++)
  {
    workspace[(tid + 1) * n + i] = in_out[i];
  }

#pragma omp barrier
#pragma omp master
  {
    for (i = 0; i < n; i++)
    {
      workspace[i] = 0;
    }

    int t;
    // assuming n is not so big, we don't parallelize this loop
    for (t = 1; t < nthreads; t++)
    {
      for (i = 0; i < n; i++)
      {
        workspace[(t + 1) * n + i] += workspace[t * n + i];
      }
    }

    for (i = 0; i < n; i++)
    {
      sum[i] = workspace[nthreads * n + i];
    }
  }
#pragma omp barrier

  for (i = 0; i < n; i++)
  {
    in_out[i] = workspace[tid * n + i];
  }
}

/**
 * sort key-value array using quick sort alogorithm, default is ascnding order 
 */
template <typename K, typename V>
void quicksort_pair(K *idx, V *w, int left, int right)
{
  if (left >= right)
    return;

  std::swap(idx[left], idx[left + (right - left) / 2]);
  std::swap(w[left], w[left + (right - left) / 2]);

  int last = left;
  for (int i = left + 1; i <= right; i++)
  {
    if (idx[i] < idx[left])
    {
      ++last;
      std::swap(idx[last], idx[i]);
      std::swap(w[last], w[i]);
    }
  }

  std::swap(idx[left], idx[last]);
  std::swap(w[left], w[last]);

  quicksort_pair(idx, w, left, last - 1);
  quicksort_pair(idx, w, last + 1, right);
}

/**
 * sort key-value array using quick sort alogorithm,
 * the parameter 'com' is a Comparable Functor to decide element order 
 * if you want to sort them in ascending order by idx value, please use std::less<int>(), otherwise use std::greater<int>()
 */
template <typename K, typename V, typename Compare>
void quicksort_pair(K *idx, V *w, int left, int right, Compare com)
{
  if (left >= right)
    return;

  std::swap(idx[left], idx[left + (right - left) / 2]);
  std::swap(w[left], w[left + (right - left) / 2]);

  int last = left;
  for (int i = left + 1; i <= right; i++)
  {
    if (com(idx[i] , idx[left]) )
    {
      ++last;
      std::swap(idx[last], idx[i]);
      std::swap(w[last], w[i]);
    }
  }

  std::swap(idx[left], idx[last]);
  std::swap(w[left], w[last]);

  quicksort_pair(idx, w, left, last - 1, com);
  quicksort_pair(idx, w, last + 1, right, com);
}
// void quicksort(int *idx, double *w, int left, int right)
// {

// }

template <typename T>
std::vector<T> calculateQuartiles(const T *data, const int len)
{
  std::vector<T> quartiles(3);
  std::vector<T> vec(data, data + len); // pointer initialize

  std::sort(vec.begin(), vec.end());
  if (len % 2 == 0) {
        quartiles[1] = (vec[len/2 - 1] + vec[len/2]) / 2.0;
    } else {
        quartiles[1] = vec[len/2];
    }
    
    // 计算下四分位数(Q1)
    size_t q1_pos = len / 4;
    if (len % 4 == 0 || len % 4 == 1) {
        quartiles[0] = (vec[q1_pos - 1] + vec[q1_pos]) / 2.0;
    } else {
        quartiles[0] = vec[q1_pos];
    }
    
    // 计算上四分位数(Q3)
    size_t q3_pos = 3 * len / 4;
    if (3 * len % 4 == 0 || 3 * len % 4 == 1) {
        quartiles[2] = (vec[q3_pos - 1] + vec[q3_pos]) / 2.0;
    } else {
        quartiles[2] = vec[q3_pos];
    }
    return quartiles;

}
