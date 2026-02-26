#pragma once

#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <iostream>
#include <cmath>

// using namespace std

// #define MALLOC(type, len) {(type *) malloc(sizeof(type) * len)}
// #define FREE(pointer) {if(pointer) free(pointer); pointer = NULL;}

// template<class T>
// void copyVector(T *out, const T *in, int len);

// template<class T>
// void vectorInit(T *array, T value, int len);

// template <class T>
// void vectorExclusiveScan(T *array, int left, int right);

template <class T>
void copyVector(T *dst, const T *src, int len)
{
#pragma omp parallel for
  for (int i = 0; i < len; ++i)
  {
    dst[i] = src[i];
  }
}

template <class T>
void vectorInit(T *array, T value, int len)
{
  // if(len < 0)
#pragma omp parallel for
  for (int i = 0; i < len; ++i)
  {
    array[i] = value;
  }
}

/** exclusive scan for [left, right] */
template <class T>
void vectorExclusiveScan(T *array, int left, int right)
{
  int length = right - left + 1;
  if (length <= 1)
    return;

  int old_val, new_val;

  old_val = array[left];
  array[left] = 0;
  for (int i = left + 1; i <= right; i++)
  {
    new_val = array[i];
    array[i] = old_val + array[i - 1];
    old_val = new_val;
  }
  // for (int i = left; i < right; i++)
  // {
  //   array[i + 1] = array[]
  // }
}
template <typename T>
bool ivectorEqual(T *first, T *second, int len)
{
  int flag = true;
  if (first && second)
  {
    for (int i = 0; i < len; i++)
    {
      if (first[i] != second[i])
      {
        flag = false;
        break;
      }
    }
    return flag;
  }
  else if (!first && !first)
  {
    return true;
  }
  else
  {
    return false;
  }
}

template <typename T>
bool fvectorEqual(T *first, T *second, int len, double tolerance)
{
  bool flag = true;
  if (first && second)
  {
    for (int i = 0; i < len; i++)
    {
      if (std::fabs(first[i] - second[i]) > tolerance)
      {
        printf("%d, first is %.16f, sencond is %.16f,  error is %.16f\n", i, first[i], second[i], std::fabs(first[i] - second[i]));
        flag = false;
        break;
      }
    }
    return flag;
  }
  else if (!first && !first)
  {
    return true;
  }
  else
  {
    return false;
  }
}

template <typename T>
void vectorReorder(T *src, T *tmp, int *perm, int len)
{
  if (!perm || !src || !tmp)
    return;

  std::copy(src, src + len, tmp);
#pragma omp parall for
  for (int i = 0; i < len; i++)
  {
    src[perm[i]] = tmp[i];
  }
}

/**
 * @note must be called inside an omp region
 */
void getSimpleThreadPartition(int *begin, int *end, int n);

/**
 * Get a load balanced partition so that each thread can work on
 * the range of begin-end where prefixSum[end] - prefixSum[begin]
 * is similar among threads.
 * For example, prefixSum can be rowptr of a CSR matrix and n can be
 * the number of rows. Then, each thread will work on similar number
 * of non-zeros.
 *
 * @params prefixSum monotonically increasing array with length n + 1
 *
 * @note must be called inside an omp region
 */
void getLoadBalancedPartition(int *begin, int *end, const int *prefixSum, int n);

void getInversePerm(int *inversePerm, const int *perm, int n);

bool isPerm(const int *perm, int n);

bool isInversePerm(const int *perm, const int *inversePerm, int len);