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

#include "solver.h"

/******************************************************************************/

typedef struct
{
  double t0, te, tol;

  double *y, *y0, **v;
  double *err, *dy;

  double **A, *b, *b_hat, *c;
  int s, ord;

  int *elem_offset, *elem_length;

  barrier_t barrier;
  reduction_t reduction;

} shared_arg_t;

/******************************************************************************/

typedef struct
{
  int me;
  shared_arg_t *shared;
} arg_t;

/******************************************************************************/

void *solver_thread(void *argument)
{
  double **v, *w, *y, *y0, *y_old, *err, *dy;
  double **A, *b, *b_hat, *c;
  double timer, err_max, h, t, tol, t0, te;
  int i, j, l, s, ord, first_elem, last_elem, num_elems, me;
  int steps_acc = 0, steps_rej = 0;
  barrier_t *bar;
  reduction_t *red;
  shared_arg_t *shared;

  me = ((arg_t *) argument)->me;
  shared = ((arg_t *) argument)->shared;

  t0 = shared->t0;
  te = shared->te;
  tol = shared->tol;

  y0 = shared->y0;
  y = shared->y;
  v = shared->v;
  err = shared->err;
  dy = shared->dy;

  A = shared->A;
  b = shared->b;
  b_hat = shared->b_hat;
  c = shared->c;

  s = shared->s;
  ord = shared->ord;

  bar = &shared->barrier;
  red = &shared->reduction;

  first_elem = shared->elem_offset[me];
  num_elems = shared->elem_length[me];
  last_elem = first_elem + num_elems - 1;

  w = y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y + first_elem, y0 + first_elem, num_elems);

  barrier_wait(bar);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    for (j = first_elem; j <= last_elem; j++)
      v[0][j] = h * ode_eval_comp(j, t + c[0] * h, y);

    for (l = 1; l < s; l++)
    {
      for (j = first_elem; j <= last_elem; j++)
        w[j] = y[j] + A[l][0] * v[0][j];

      for (i = 1; i < l; i++)
        for (j = first_elem; j <= last_elem; j++)
          w[j] += A[l][i] * v[i][j];

      barrier_wait(bar);

      for (j = first_elem; j <= last_elem; j++)
        v[l][j] = h * ode_eval_comp(j, t + c[l] * h, w);

      barrier_wait(bar);
    }

    for (j = first_elem; j <= last_elem; j++)
    {
      err[j] = b_hat[0] * v[0][j];
      dy[j] = b[0] * v[0][j];
    }

    for (i = 1; i < s; i++)
      for (j = first_elem; j <= last_elem; j++)
      {
        err[j] += b_hat[i] * v[i][j];
        dy[j] += b[i] * v[i][j];
      }

    err_max = 0.0;
    for (j = first_elem; j <= last_elem; j++)
    {
      double yj_old = y[j];
      y[j] += dy[j];
      y_old[j] = yj_old;        /* y_old and dy occupy the same space */
      update_error_max(&err_max, err[j], y[j], yj_old);
    }

    err_max = reduction_max(red, err_max);

    /* step control */

    step_control(&t, &h, err_max, ord, tol, y + first_elem, y_old + first_elem,
                 num_elems, &steps_acc, &steps_rej);

    barrier_wait(bar);
  }

  timer_stop(&timer);

  if (me == 0)
    print_statistics(timer, steps_acc, steps_rej);

  return NULL;
}

/******************************************************************************/

void solver(double t0, double te, double *y0, double *y, double tol)
{
  arg_t *arg;
  shared_arg_t *shared;
  void **arglist;
  double **A, *b, *b_hat, *c;
  int i, s, ord;

  printf("Solver type: ");
  printf("parallel embedded Runge-Kutta method for shared address space\n");
  printf("Implementation variant: A (spatial locality)\n");
  printf("Number of threads: %d\n", threads);

  arg = MALLOC(threads, arg_t);
  shared = MALLOC(1, shared_arg_t);
  arglist = MALLOC(threads, void *);

  shared->y0 = y0;
  shared->y = y;

  shared->t0 = t0;
  shared->te = te;
  shared->tol = tol;

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  shared->A = A;
  shared->b = b;
  shared->c = c;

  shared->b_hat = MALLOC(s, double);
  for (i = 0; i < s; i++)
    shared->b_hat[i] = b[i] - b_hat[i];

  shared->s = s;
  shared->ord = ord;

  ALLOC2D(shared->v, s, ode_size, double);
  shared->err = MALLOC(ode_size, double);
  shared->dy = MALLOC(ode_size, double);

  barrier_init(&shared->barrier, threads);
  reduction_init(&shared->reduction, threads);

  shared->elem_offset = MALLOC(threads, int);
  shared->elem_length = MALLOC(threads, int);

  blockwise_distribution(threads, ode_size, shared->elem_offset,
                         shared->elem_length);

  for (i = 0; i < threads; i++)
  {
    arg[i].me = i;
    arg[i].shared = shared;
    arglist[i] = (void *) (arg + i);
  }

  run_threads(threads, solver_thread, arglist);

  FREE(shared->elem_offset);
  FREE(shared->elem_length);

  barrier_destroy(&shared->barrier);
  reduction_destroy(&shared->reduction);

  free_emb_rk_method(&A, &b, &b_hat, &c, s);
  FREE(shared->b_hat);

  FREE2D(shared->v);
  FREE(shared->err);
  FREE(shared->dy);

  FREE(shared);
  FREE(arg);
  FREE(arglist);
}

/******************************************************************************/
