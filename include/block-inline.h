/******************************************************************************/
/* This file is part of a collection of embedded Runge-Kutta solvers.         */
/* Copyright (C) 2009-2020, Matthias Korch, University of Bayreuth, Germany.  */
/*                                                                            */
/* This program is free software: you can redistribute it and/or modify       */
/* it under the terms of the GNU General Public License as published by       */
/* the Free Software Foundation, either version 3 of the License, or          */
/* (at your option) any later version.                                        */
/*                                                                            */
/* This program is distributed in the hope that it will be useful,            */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of             */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              */
/* GNU General Public License for more details.                               */
/*                                                                            */
/* You should have received a copy of the GNU General Public License          */
/* along with this program.  If not, see <https://www.gnu.org/licenses/>.     */
/******************************************************************************/

#ifndef BLOCK_INLINE_H_
#define BLOCK_INLINE_H_

/******************************************************************************/

#include "config.h"
#include "tools.h"
#include "ode.h"

/******************************************************************************/

static inline void block_rhs(int l, int first, int size, double t, double h,
                             double *hc, double *w, double **v)
{
  ode_eval_rng(first, first + size - 1, t + hc[l], w, v[l]);
}

/******************************************************************************/

static inline void block_gather_interm_stage(int l, int first, int size,
                                             double **hA, double *y,
                                             double *w, double **v)
{
  int i, j;

  for (j = first; j < first + size; j++)
  {
    w[j] = y[j];

    for (i = 0; i < l; i++)
      w[j] += hA[l][i] * v[i][j];
  }
}

/******************************************************************************/

static inline void block_gather_output(int first, int size, int s,
                                       double *hb, double *hb_hat,
                                       double *err, double *dy, double **v)
{
  int i, j;

  for (j = first; j < first + size; j++)
  {
    err[j] = hb_hat[0] * v[0][j];
    dy[j] = hb[0] * v[0][j];

    for (i = 1; i < s; i++)
    {
      err[j] += hb_hat[i] * v[i][j];
      dy[j] += hb[i] * v[i][j];
    }
  }
}

/******************************************************************************/

static inline void block_gather_vec_interm_stage(int l, int first, int size,
                                                 double **hA, int **iz_A,
                                                 double *y, double *w,
                                                 double **v)
{
  int i, j;

  if (iz_A[l][0])
    for (j = first; j < first + size; j++)
      w[j] = y[j] + hA[l][0] * v[0][j];
  else
    for (j = first; j < first + size; j++)
      w[j] = y[j];

  for (i = 1; i < l; i++)
    if (iz_A[l][i])
      for (j = first; j < first + size; j++)
        w[j] += hA[l][i] * v[i][j];
}

/******************************************************************************/

static inline void block_gather_vec_output(int first, int size, int s,
                                           double *hb, double *hb_hat,
                                           int *iz_b, int *iz_b_hat,
                                           double *err, double *dy, double **v)
{
  int i, j;

  for (j = first; j < first + size; j++)
  {
    err[j] = hb_hat[0] * v[0][j];
    dy[j] = hb[0] * v[0][j];
  }

  for (i = 1; i < s; i++)
    if (iz_b_hat[i])
    {
      if (iz_b[i])
      {
        for (j = first; j < first + size; j++)
        {
          err[j] += hb_hat[i] * v[i][j];
          dy[j] += hb[i] * v[i][j];
        }
      }
      else
      {
        for (j = first; j < first + size; j++)
          err[j] += hb_hat[i] * v[i][j];
      }
    }
    else
    {
      if (iz_b[i])
        for (j = first; j < first + size; j++)
          dy[j] += hb[i] * v[i][j];
    }
}

/******************************************************************************/

static inline void tiled_block_gather_interm_stage(int l, int first, int size,
                                                   double **hA, int **iz_A,
                                                   double *y, double *w,
                                                   double **v)
{
  int i, j, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    if (iz_A[l][0])
      for (jj = 0; jj < count; jj++)
        w[j + jj] = y[j + jj] + hA[l][0] * v[0][j + jj];
    else
      for (jj = 0; jj < count; jj++)
        w[j + jj] = y[j + jj];

    for (i = 1; i < l; i++)
      if (iz_A[l][i])
        for (jj = 0; jj < count; jj++)
          w[j + jj] += hA[l][i] * v[i][j + jj];
  }
}

/******************************************************************************/

static inline void block_rhs_gather_interm_stage(int l, int first,
                                                 int size, double t,
                                                 double h, double **hA,
                                                 double *hc, double *y,
                                                 double *w_l,
                                                 double *w_lp1, double **v)
{
  int i, j;

  for (j = first; j < first + size; j++)
  {
    v[l][j] = ode_eval_comp(j, t + hc[l], w_l);

    w_lp1[j] = y[j];
    for (i = 0; i < l + 1; i++)
      w_lp1[j] += hA[l + 1][i] * v[i][j];
  }
}

/******************************************************************************/

static inline void tiled_block_rhs_gather_interm_stage(int l, int first,
                                                       int size, double t,
                                                       double h,
                                                       double **hA, int **iz_A,
                                                       double *hc,
                                                       double *y,
                                                       double *w_l,
                                                       double *w_lp1,
                                                       double **v)
{
  int i, j, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    ode_eval_rng(j, j + count - 1, t + hc[l], w_l, v[l]);

    if (iz_A[l + 1][0])
      for (jj = 0; jj < count; jj++)
        w_lp1[j + jj] = y[j + jj] + hA[l + 1][0] * v[0][j + jj];
    else
      for (jj = 0; jj < count; jj++)
        w_lp1[j + jj] = y[j + jj];

    for (i = 1; i < l + 1; i++)
      if (iz_A[l + 1][i])
        for (jj = 0; jj < count; jj++)
          w_lp1[j + jj] += hA[l + 1][i] * v[i][j + jj];
  }
}

/******************************************************************************/

static inline void tiled_block_gather_output(int first, int size, int s,
                                             double *hb, double *hb_hat,
                                             int *iz_b, int *iz_b_hat,
                                             double *err, double *dy,
                                             double **v)
{
  int i, j, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    for (jj = 0; jj < count; jj++)
      err[j + jj] = hb_hat[0] * v[0][j + jj];

    for (jj = 0; jj < count; jj++)
      dy[j + jj] = hb[0] * v[0][j + jj];

    for (i = 1; i < s; i++)
    {
      if (iz_b_hat[i])
        for (jj = 0; jj < count; jj++)
          err[j + jj] += hb_hat[i] * v[i][j + jj];

      if (iz_b[i])
        for (jj = 0; jj < count; jj++)
          dy[j + jj] += hb[i] * v[i][j + jj];
    }
  }
}

/******************************************************************************/

static inline void block_scatter_first_stage(int first, int size, int s,
                                             double t, double h, double **hA,
                                             double *hb, double *hb_hat,
                                             double *hc, double *y, double *err,
                                             double *dy, double **w)
{
  int j, l;

  for (j = first; j < first + size; j++)
  {
    double F = ode_eval_comp(j, t + hc[0], y);
    double Y = y[j];

    for (l = 1; l < s; l++)
      w[l][j] = Y + hA[l][0] * F;

    dy[j] = hb[0] * F;
    err[j] = hb_hat[0] * F;
  }
}

/******************************************************************************/

static inline void block_scatter_interm_stage(int i, int first, int size, int s,
                                              double t, double h, double **hA,
                                              double *hb, double *hb_hat,
                                              double *c, double *y, double *err,
                                              double *dy, double **w)
{
  int j, l;

  for (j = first; j < first + size; j++)
  {
    double F = ode_eval_comp(j, t + c[i], w[i]);

    for (l = i + 1; l < s; l++)
      w[l][j] += hA[l][i] * F;

    dy[j] += hb[i] * F;
    err[j] += hb_hat[i] * F;
  }
}

/******************************************************************************/

static inline void block_scatter_last_stage(int first, int size, int s,
                                            double t, double h, double *hb,
                                            double *hb_hat, double *hc,
                                            double *y, double *err, double *dy,
                                            double **w, double *err_max)
{
  int j, i = s - 1;

  for (j = first; j < first + size; j++)
  {
    double yj_old;
    double F = ode_eval_comp(j, t + hc[i], w[i]);

    dy[j] += hb[i] * F;
    err[j] += hb_hat[i] * F;

    yj_old = y[j];
    y[j] += dy[j];
    dy[j] = yj_old;             /* y_old and dy occupy the same space */

    update_error_max(err_max, err[j], y[j], yj_old);
  }
}

/******************************************************************************/

static inline void tiled_block_scatter_first_stage(int first, int size, int s,
                                                   double t, double h,
                                                   double **hA, int **iz_A,
                                                   double *hb, double *hb_hat,
                                                   double *hc, double *y,
                                                   double *err, double *dy,
                                                   double **w, double *v)
{
  int j, l, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    ode_eval_rng(j, j + count - 1, t + hc[0], y, v - j);

    for (l = 1; l < s; l++)
    {
      if (iz_A[l][0])
        for (jj = 0; jj < count; jj++)
          w[l][j + jj] = y[j + jj] + hA[l][0] * v[jj];
      else
        for (jj = 0; jj < count; jj++)
          w[l][j + jj] = y[j + jj];
    }

    for (jj = 0; jj < count; jj++)
      dy[j + jj] = hb[0] * v[jj];

    for (jj = 0; jj < count; jj++)
      err[j + jj] = hb_hat[0] * v[jj];
  }
}

/******************************************************************************/

static inline void tiled_block_scatter_interm_stage(int i, int first, int size,
                                                    int s, double t, double h,
                                                    double **hA, double *hb,
                                                    double *hb_hat, double *hc,
                                                    int **iz_A, int *iz_b,
                                                    int *iz_b_hat, double *y,
                                                    double *err, double *dy,
                                                    double **w, double *v)
{
  int j, l, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    ode_eval_rng(j, j + count - 1, t + hc[i], w[i], v - j);

    for (l = i + 1; l < s; l++)
      if (iz_A[l][i])
        for (jj = 0; jj < count; jj++)
          w[l][j + jj] += hA[l][i] * v[jj];

    if (iz_b[i])
      for (jj = 0; jj < count; jj++)
        dy[j + jj] += hb[i] * v[jj];

    if (iz_b_hat[i])
      for (jj = 0; jj < count; jj++)
        err[j + jj] += hb_hat[i] * v[jj];
  }
}

/******************************************************************************/

static inline void tiled_block_scatter_last_stage(int first, int size, int s,
                                                  double t, double h,
                                                  double *hb, double *hb_hat,
                                                  double *hc, int *iz_b,
                                                  int *iz_b_hat, double *y,
                                                  double *err, double *dy,
                                                  double **w, double *v,
                                                  double *err_max)
{
  int j, i = s - 1, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    ode_eval_rng(j, j + count - 1, t + hc[i], w[i], v - j);

    if (iz_b[i])
      for (jj = 0; jj < count; jj++)
        dy[j + jj] += hb[i] * v[jj];

    if (iz_b_hat[i])
      for (jj = 0; jj < count; jj++)
        err[j + jj] += hb_hat[i] * v[jj];

    for (jj = 0; jj < count; jj++)
    {
      double yj_old = y[j + jj];
      y[j + jj] += dy[j + jj];
      dy[j + jj] = yj_old;      /* y_old and dy occupy the same space */

      update_error_max(err_max, err[j + jj], y[j + jj], yj_old);
    }
  }
}

/******************************************************************************/

static inline void init_mutexes(int me, int s, mutex_lock_t ** mutex_first,
                                mutex_lock_t ** mutex_last)
{
  int j;

  for (j = 1; j < s; j++)
  {
    mutex_lock_lock(&(mutex_first[me][j]));
    mutex_lock_lock(&(mutex_last[me][j]));
  }
}

/******************************************************************************/

static inline void wait_for_pred(int me, int i, mutex_lock_t ** mutex)
{
  if (me > 0)
    mutex_lock_lock(&(mutex[me - 1][i]));
}

/******************************************************************************/

static inline void wait_for_succ(int me, int i, mutex_lock_t ** mutex)
{
  if (me < threads - 1)
    mutex_lock_lock(&(mutex[me + 1][i]));
}

/******************************************************************************/

static inline void release_pred(int me, int i, mutex_lock_t ** mutex)
{
  if (me > 0)
    mutex_lock_unlock(&(mutex[me - 1][i]));
}

/******************************************************************************/

static inline void release_succ(int me, int i, mutex_lock_t ** mutex)
{
  if (me < threads - 1)
    mutex_lock_unlock(&(mutex[me + 1][i]));
}

/******************************************************************************/

static inline void first_block_complete(int me, int i, mutex_lock_t ** mutex)
{
  mutex_lock_unlock(&(mutex[me][i]));
}

/******************************************************************************/

static inline void last_block_complete(int me, int i, mutex_lock_t ** mutex)
{
  mutex_lock_unlock(&(mutex[me][i]));
}

/******************************************************************************/

static inline void lock_init_phase(uint me, mutex_lock_t ** mutex)
{
  mutex_lock_lock(&(mutex[me][0]));
}

/******************************************************************************/

static inline void unlock_init_phase(uint me, mutex_lock_t ** mutex)
{
  mutex_lock_unlock(&(mutex[me][0]));
}

/******************************************************************************/

static inline void wait_pred_init_complete(uint me, mutex_lock_t ** mutex)
{
  if (me > 0)
  {
    mutex_lock_lock(&(mutex[me - 1][0]));
    mutex_lock_unlock(&(mutex[me - 1][0]));
  }
}

/******************************************************************************/

static inline void wait_succ_init_complete(uint me, mutex_lock_t ** mutex)
{
  if (me < threads - 1)
  {
    mutex_lock_lock(&(mutex[me + 1][0]));
    mutex_lock_unlock(&(mutex[me + 1][0]));
  }
}

/******************************************************************************/

static inline void get_from_pred(uint me, uint i, double **my_w,
                                 double **nb_w, uint first, uint B,
                                 mutex_lock_t ** mutex)
{
  if (me > 0)
  {
    mutex_lock_lock(&(mutex[me - 1][i]));
    mutex_lock_unlock(&(mutex[me - 1][i]));
    copy_vector(my_w[i] + first - B, nb_w[i] + first - B, B);
  }
}

/******************************************************************************/

static inline void get_from_succ(uint me, uint i, double **my_w,
                                 double **nb_w, uint last, uint B,
                                 mutex_lock_t ** mutex)
{
  if (me < threads - 1)
  {
    mutex_lock_lock(&(mutex[me + 1][i]));
    mutex_lock_unlock(&(mutex[me + 1][i]));
    copy_vector(my_w[i] + last + 1, nb_w[i] + last + 1, B);
  }
}

/******************************************************************************/

#ifdef HAVE_MPI

/******************************************************************************/

static inline void start_send_pred(double *w, int first, int B, int tag,
                                   MPI_Request * request)
{
  if (me > 0)
    MPI_Isend(w + first, B, MPI_DOUBLE, me - 1, tag, MPI_COMM_WORLD, request);
}

/******************************************************************************/

static inline void start_send_succ(double *w, int last, int B, int tag,
                                   MPI_Request * request)
{
  if (me < processes - 1)
    MPI_Isend(w + last - B + 1, B, MPI_DOUBLE, me + 1, tag,
              MPI_COMM_WORLD, request);
}

/******************************************************************************/

static inline void start_recv_pred(double *w, int first, int B, int tag,
                                   MPI_Request * request)
{
  if (me > 0)
    MPI_Irecv(w + first - B, B, MPI_DOUBLE, me - 1, tag,
              MPI_COMM_WORLD, request);
}

/******************************************************************************/

static inline void start_recv_succ(double *w, int last, int B, int tag,
                                   MPI_Request * request)
{
  if (me < processes - 1)
    MPI_Irecv(w + last + 1, B, MPI_DOUBLE, me + 1, tag, MPI_COMM_WORLD,
              request);
}

/******************************************************************************/

static inline void complete_send_pred(MPI_Request * request,
                                      MPI_Status * status)
{
  if (me > 0)
    MPI_Wait(request, status);
}

/******************************************************************************/

static inline void complete_send_succ(MPI_Request * request,
                                      MPI_Status * status)
{
  if (me < processes - 1)
    MPI_Wait(request, status);
}

/******************************************************************************/

static inline void complete_recv_pred(MPI_Request * request,
                                      MPI_Status * status)
{
  if (me > 0)
    MPI_Wait(request, status);
}

/******************************************************************************/

static inline void complete_recv_succ(MPI_Request * request,
                                      MPI_Status * status)
{
  if (me < processes - 1)
    MPI_Wait(request, status);
}

/******************************************************************************/

#endif /* HAVE_MPI */

/******************************************************************************/

#endif /* BLOCK_INLINE */

/******************************************************************************/
