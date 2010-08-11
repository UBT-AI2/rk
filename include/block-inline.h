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

#ifndef BLOCK_INLINE_H_
#define BLOCK_INLINE_H_

/******************************************************************************/

#include "config.h"
#include "tools.h"
#include "ode.h"

/******************************************************************************/

static inline void block_first_stage(int first, int size, int s, double t,
                                     double h, double **A, double *b,
                                     double *b_hat, double *c, double *y,
                                     double *err, double *dy, double **w)
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

static inline void block_first_stage_reverse(int first, int size, int s,
                                             double t, double h, double **A,
                                             double *b, double *b_hat,
                                             double *c, double *y, double *err,
                                             double *dy, double **w)
{
  int j, l;

  for (j = first + size - 1; j >= first; j--)
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

static inline void block_interm_stage(int i, int first, int size, int s,
                                      double t, double h, double **A, double *b,
                                      double *b_hat, double *c, double *y,
                                      double *err, double *dy, double **w)
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

static inline void block_interm_stage_reverse(int i, int first, int size, int s,
                                              double t, double h, double **A,
                                              double *b, double *b_hat,
                                              double *c, double *y, double *err,
                                              double *dy, double **w)
{
  int j, l;

  for (j = first + size - 1; j >= first; j--)
  {
    double hF = h * ode_eval_comp(j, t + c[i] * h, w[i]);

    for (l = i + 1; l < s; l++)
      w[l][j] += A[l][i] * hF;

    dy[j] += b[i] * hF;
    err[j] += b_hat[i] * hF;
  }
}

/******************************************************************************/

static inline void block_last_stage(int first, int size, int s,
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

static inline void block_last_stage_reverse(int first, int size, int s,
                                            double t, double h, double *b,
                                            double *b_hat, double *c, double *y,
                                            double *err, double *dy,
                                            double **w, double *err_max)
{
  int j, i = s - 1;

  for (j = first + size - 1; j >= first; j--)
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
