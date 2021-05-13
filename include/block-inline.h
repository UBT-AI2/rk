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
                             double *c, double *w, double **v)
{
  int j;

  ode_eval_rng(first, first + size - 1, t + c[l] * h, w, v[l]);

  for (j = first; j < first + size; j++)
    v[l][j] *= h;
}

/******************************************************************************/

static inline void block_gather_interm_stage(int l, int first, int size,
                                             double **A, double *y, double *w,
                                             double **v)
{
  int i, j;

  for (j = first; j < first + size; j++)
  {
    w[j] = y[j];

    for (i = 0; i < l; i++)
      w[j] += A[l][i] * v[i][j];
  }
}

/******************************************************************************/

static inline void block_gather_output(int first, int size, int s,
                                       double *b, double *b_hat,
                                       double *err, double *dy, double **v)
{
  int i, j;

  for (j = first; j < first + size; j++)
  {
    err[j] = b_hat[0] * v[0][j];
    dy[j] = b[0] * v[0][j];

    for (i = 1; i < s; i++)
    {
      err[j] += b_hat[i] * v[i][j];
      dy[j] += b[i] * v[i][j];
    }
  }
}

/******************************************************************************/

static inline void block_gather_vec_interm_stage(int l, int first, int size,
                                                 double **A, double *y,
                                                 double *w, double **v)
{
  int i, j;

  if (A[l][0] != 0.0)
    for (j = first; j < first + size; j++)
      w[j] = y[j] + A[l][0] * v[0][j];
  else
    for (j = first; j < first + size; j++)
      w[j] = y[j];

  for (i = 1; i < l; i++)
    if (A[l][i] != 0.0)
      for (j = first; j < first + size; j++)
        w[j] += A[l][i] * v[i][j];
}

/******************************************************************************/

static inline void block_gather_vec_output(int first, int size, int s,
                                           double *b, double *b_hat,
                                           double *err, double *dy, double **v)
{
  int i, j;

  for (j = first; j < first + size; j++)
  {
    err[j] = b_hat[0] * v[0][j];
    dy[j] = b[0] * v[0][j];
  }

  for (i = 1; i < s; i++)
    if (b_hat[i] != 0.0)
    {
      if (b[i] != 0.0)
      {
        for (j = first; j < first + size; j++)
        {
          err[j] += b_hat[i] * v[i][j];
          dy[j] += b[i] * v[i][j];
        }
      }
      else
      {
        for (j = first; j < first + size; j++)
          err[j] += b_hat[i] * v[i][j];
      }
    }
    else
    {
      if (b[i] != 0.0)
        for (j = first; j < first + size; j++)
          dy[j] += b[i] * v[i][j];
    }
}

/******************************************************************************/

static inline void tiled_block_gather_interm_stage(int l, int first, int size,
                                                   double **A, double *y,
                                                   double *w, double **v)
{
  int i, j, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    if (A[l][0] != 0.0)
      for (jj = 0; jj < count; jj++)
        w[j + jj] = y[j + jj] + A[l][0] * v[0][j + jj];
    else
      for (jj = 0; jj < count; jj++)
        w[j + jj] = y[j + jj];

    for (i = 1; i < l; i++)
      if (A[l][i] != 0.0)
        for (jj = 0; jj < count; jj++)
          w[j + jj] += A[l][i] * v[i][j + jj];
  }
}

/******************************************************************************/

static inline void block_rhs_gather_interm_stage(int l, int first,
                                                 int size, double t,
                                                 double h, double **A,
                                                 double *c, double *y,
                                                 double *w_l,
                                                 double *w_lp1, double **v)
{
  int i, j;

  for (j = first; j < first + size; j++)
  {
    v[l][j] = h * ode_eval_comp(j, t + c[l] * h, w_l);

    w_lp1[j] = y[j];
    for (i = 0; i < l + 1; i++)
      w_lp1[j] += A[l + 1][i] * v[i][j];
  }
}

/******************************************************************************/

static inline void tiled_block_rhs_gather_interm_stage(int l, int first,
                                                       int size, double t,
                                                       double h,
                                                       double **A,
                                                       double *c,
                                                       double *y,
                                                       double *w_l,
                                                       double *w_lp1,
                                                       double **v)
{
  int i, j, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    ode_eval_rng(j, j + count - 1, t + c[l] * h, w_l, v[l]);
    for (jj = 0; jj < count; jj++)
      v[l][j + jj] *= h;

    for (jj = 0; jj < count; jj++)
      w_lp1[j + jj] = y[j + jj] + A[l + 1][0] * v[0][j + jj];

    for (i = 1; i < l + 1; i++)
      for (jj = 0; jj < count; jj++)
        w_lp1[j + jj] += A[l + 1][i] * v[i][j + jj];
  }
}

/******************************************************************************/

static inline void tiled_block_gather_output(int first, int size, int s,
                                             double *b, double *b_hat,
                                             double *err, double *dy,
                                             double **v)
{
  int i, j, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    for (jj = 0; jj < count; jj++)
      err[j + jj] = b_hat[0] * v[0][j + jj];

    for (jj = 0; jj < count; jj++)
      dy[j + jj] = b[0] * v[0][j + jj];

    for (i = 1; i < s; i++)
    {
      if (b_hat[i] != 0.0)
        for (jj = 0; jj < count; jj++)
          err[j + jj] += b_hat[i] * v[i][j + jj];

      if (b[i] != 0.0)
        for (jj = 0; jj < count; jj++)
          dy[j + jj] += b[i] * v[i][j + jj];
    }
  }
}

/******************************************************************************/

static inline void block_scatter_first_stage(int first, int size, int s,
                                             double t, double h, double **A,
                                             double *b, double *b_hat,
                                             double *c, double *y, double *err,
                                             double *dy, double **w)
{
  int j, l;

  for (j = first; j < first + size; j++)
  {
    double hF = h * ode_eval_comp(j, t + c[0] * h, y);
    double Y = y[j];

    for (l = 1; l < s; l++)
      w[l][j] = Y + A[l][0] * hF;

    dy[j] = b[0] * hF;
    err[j] = b_hat[0] * hF;
  }
}

/******************************************************************************/

static inline void block_scatter_interm_stage(int i, int first, int size, int s,
                                              double t, double h, double **A,
                                              double *b, double *b_hat,
                                              double *c, double *y, double *err,
                                              double *dy, double **w)
{
  int j, l;

  for (j = first; j < first + size; j++)
  {
    double hF = h * ode_eval_comp(j, t + c[i] * h, w[i]);

    for (l = i + 1; l < s; l++)
      w[l][j] += A[l][i] * hF;

    dy[j] += b[i] * hF;
    err[j] += b_hat[i] * hF;
  }
}

/******************************************************************************/

static inline void block_scatter_last_stage(int first, int size, int s,
                                            double t, double h, double *b,
                                            double *b_hat, double *c, double *y,
                                            double *err, double *dy,
                                            double **w, double *err_max)
{
  int j, i = s - 1;

  for (j = first; j < first + size; j++)
  {
    double yj_old;
    double hF = h * ode_eval_comp(j, t + c[i] * h, w[i]);

    dy[j] += b[i] * hF;
    err[j] += b_hat[i] * hF;

    yj_old = y[j];
    y[j] += dy[j];
    dy[j] = yj_old;             /* y_old and dy occupy the same space */

    update_error_max(err_max, err[j], y[j], yj_old);
  }
}

/******************************************************************************/

static inline void tiled_block_scatter_first_stage(int first, int size, int s,
                                                   double t, double h,
                                                   double **A, double *b,
                                                   double *b_hat, double *c,
                                                   double *y, double *err,
                                                   double *dy, double **w,
                                                   double *v)
{
  int j, l, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    ode_eval_rng(j, j + count - 1, t + c[0] * h, y, v - j);
    for (jj = 0; jj < count; jj++)
      v[jj] *= h;

    for (l = 1; l < s; l++)
    {
      if (A[l][0] != 0.0)
        for (jj = 0; jj < count; jj++)
          w[l][j + jj] = y[j + jj] + A[l][0] * v[jj];
      else
        for (jj = 0; jj < count; jj++)
          w[l][j + jj] = y[j + jj];
    }

    for (jj = 0; jj < count; jj++)
      dy[j + jj] = b[0] * v[jj];

    for (jj = 0; jj < count; jj++)
      err[j + jj] = b_hat[0] * v[jj];
  }
}

/******************************************************************************/

static inline void tiled_block_scatter_interm_stage(int i, int first, int size,
                                                    int s, double t, double h,
                                                    double **A, double *b,
                                                    double *b_hat, double *c,
                                                    double *y, double *err,
                                                    double *dy, double **w,
                                                    double *v)
{
  int j, l, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    ode_eval_rng(j, j + count - 1, t + c[i] * h, w[i], v - j);
    for (jj = 0; jj < count; jj++)
      v[jj] *= h;

    for (l = i + 1; l < s; l++)
      if (A[l][i] != 0.0)
        for (jj = 0; jj < count; jj++)
          w[l][j + jj] += A[l][i] * v[jj];

    if (b[i] != 0.0)
      for (jj = 0; jj < count; jj++)
        dy[j + jj] += b[i] * v[jj];

    if (b_hat[i] != 0.0)
      for (jj = 0; jj < count; jj++)
        err[j + jj] += b_hat[i] * v[jj];
  }
}

/******************************************************************************/

static inline void tiled_block_scatter_last_stage(int first, int size, int s,
                                                  double t, double h, double *b,
                                                  double *b_hat, double *c,
                                                  double *y, double *err,
                                                  double *dy, double **w,
                                                  double *v, double *err_max)
{
  int j, i = s - 1, jj;

  for (j = first; j < first + size; j += BLOCKSIZE)
  {
    int count = imin(BLOCKSIZE, first + size - j);

    ode_eval_rng(j, j + count - 1, t + c[i] * h, w[i], v - j);
    for (jj = 0; jj < count; jj++)
      v[jj] *= h;

    if (b[i] != 0.0)
      for (jj = 0; jj < count; jj++)
        dy[j + jj] += b[i] * v[jj];

    if (b_hat[i] != 0.0)
      for (jj = 0; jj < count; jj++)
        err[j + jj] += b_hat[i] * v[jj];

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
