/******************************************************************************/
/* This file is part of a collection of embedded Runge-Kutta solvers.         */
/* Copyright (C) 2009-2010, Matthias Korch, University of Bayreuth, Germany.  */
/*                                                                            */
/* This is free software; you can redistribute it and/or modify it under the  */
/* terms of the GNU General Public License as published by the Free Software  */
/* Foundation; either version 2 of the License, or (at your option) any later */
/* version.                                                                   */
/*                                                                            */
/* This software is distributed in the hope that it will be useful, but       */
/* WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY */
/* or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License    */
/* for more details.                                                          */
/*                                                                            */
/* You should have received a copy of the GNU General Public License along    */
/* with this program; if not, write to the Free Software Foundation, Inc.,    */
/* 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.              */
/******************************************************************************/

#ifndef TOOLS_H_
#define TOOLS_H_

/******************************************************************************/

#include "config.h"

#include <stdio.h>
#include <malloc.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

/******************************************************************************/
/* Minimum and maximum computations                                           */
/******************************************************************************/

#ifndef __USE_ISOC99

static inline double fmax(double a, double b)
{
  if (a > b)
    return a;
  else
    return b;
}

static inline double fmin(double a, double b)
{
  if (a < b)
    return a;
  else
    return b;
}

#endif

static inline int imax(int a, int b)
{
  if (a > b)
    return a;
  else
    return b;
}

static inline int imin(int a, int b)
{
  if (a < b)
    return a;
  else
    return b;
}

/******************************************************************************/
/* Swapping variables                                                         */
/******************************************************************************/

static inline void swap_matrices(double ***a, double ***b)
{
  double **tmp = *b;
  *b = *a;
  *a = tmp;
}

static inline void swap_vectors(double **a, double **b)
{
  double *tmp = *b;
  *b = *a;
  *a = tmp;
}

/******************************************************************************/
/* Memory management                                                          */
/******************************************************************************/

/* Macro to be used in the defintion of padded data structures 
 to minimize the effects of false sharing. */

#ifndef PAD_SIZE
#define PAD_SIZE             0
#endif

#define PAD_SIZE2            ((PAD_SIZE) + (PAD_SIZE))

#if PAD_SIZE > 0
#define PADDING_FIELD(id)    char id[PAD_SIZE];
#else
#define PADDING_FIELD(id)
#endif

/* MALLOC and FREE macros for 1D arrays */

#define MALLOC(n,t) \
  ((t *) (((char *) malloc((n) * sizeof(t) + PAD_SIZE2)) + PAD_SIZE))

#define FREE(p)                                   \
  do                                              \
  {                                               \
    if (p != NULL)                                \
    {                                             \
      free((void *) (((char *) (p)) - PAD_SIZE)); \
      p = NULL;                                   \
    }                                             \
  } while (0)

/* MALLOC and FREE macros for 2D arrays */

#define ALLOC2D(x,a,b,t)                       \
  do                                           \
  {                                            \
    int a2d_i_;                                \
    x = MALLOC((a) + 1, t*);                   \
    for (a2d_i_ = 0; a2d_i_ < (a); a2d_i_++)   \
      (x)[a2d_i_] = MALLOC(b, t);              \
    (x)[a] = NULL;                             \
  } while (0)

#define FREE2D(x)                 \
  do                              \
  {                               \
    int f2d_i_ = -1;              \
    while ((x)[++f2d_i_] != NULL) \
      FREE((x)[f2d_i_]);          \
    FREE(x);                      \
  } while (0)

/******************************************************************************/
/* Vector copy                                                                */
/******************************************************************************/

static inline void copy_vector(double *dst, double *src, int n)
{
#ifdef USE_MEMCPY
  memcpy(dst, src, n * sizeof(double));
#else
  int i;
  for (i = 0; i < n; i++)
    dst[i] = src[i];
#endif
}

/******************************************************************************/
/* Time measurement                                                           */
/******************************************************************************/

static inline double current_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double) tv.tv_sec + (double) tv.tv_usec * 1E-6;
}

static inline void timer_start(double *T)
{
  *T = current_time();
}

static inline void timer_stop(double *T)
{
  *T = current_time() - *T;
}

/******************************************************************************/
/* Data distribution                                                          */
/******************************************************************************/

void blockwise_distribution(int processors, int n, int *first, int *size);

/******************************************************************************/
/* Walk through integration interval                                          */
/******************************************************************************/

#if (STEP_LIMIT > 0)
#define FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej) \
  for (t = t0; (t < te) && (steps_acc + steps_rej < STEP_LIMIT); \
       h = fmin(h, te - t))
#else
#define FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej) \
  for (t = t0; t < te; h = fmin(h, te - t))
#endif

/******************************************************************************/
/* Stepsize selection                                                         */
/******************************************************************************/

double initial_stepsize(double t0, double H, const double *y_0, int ord,
                        double tol);

/*
 * Stepsize selection according to
 * 
 *   K. Strehmel and R. Weiner: Numerik gewoehnlicher Differentialgleichungen, 
 *   Teubner, 1995, eq. (2.5.13).
 * 
 *   J. C. Butcher: Numerical methods for ordinary differential equations,
 *   Wiley, 2003, eq. (391a).    
 */


#define AMAX  2.0               /* 1.5, ..., 5.0 */
#define AMIN  0.5               /* 0.2, ..., 0.5 */
#define ASAF  0.9               /* 0.8, 0.9, or pow(0.25, 1.0 / (ord) + 1.0) */

static inline double h_new_acc(double h, double err, double tol, int ord)
{
  return h * fmin((AMAX), fmax((AMIN), (ASAF) * pow(tol / err, 1.0
                                                    / (ord + 1.0))));
}

static inline double h_new_rej(double h, double err, double tol, int ord)
{
  return h_new_acc(h, err, tol, ord);
}

static inline int step_control(double *t, double *h, double err, int ord,
                               double tol, double *y, double *y_old, int n,
                               int *counter_acc, int *counter_rej)
{
  if (err <= tol)               /* accept */
  {
    *t += *h;
    *h = h_new_acc(*h, err, tol, ord);
    (*counter_acc)++;
    return 1;
  }
  else                          /* reject */
  {
    copy_vector(y, y_old, n);
    *h = h_new_rej(*h, err, tol, ord);
    (*counter_rej)++;
    return 0;
  }
}

static inline void update_error_max(double *emax, double ej, double yj,
                                    double yj_old)
{
  double divisor = fmax(fabs(yj), fabs(yj_old));
  if (divisor == 0.0)
    *emax = 1.7E308;
  else
    *emax = fmax(*emax, fabs(ej) / divisor);
}

/******************************************************************************/
/* Print statistics                                                           */
/******************************************************************************/

void print_statistics(double timer, int steps_acc, int steps_rej);

/******************************************************************************/

#endif /* TOOLS_H_ */

/******************************************************************************/
