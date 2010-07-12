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

  double *y, *y0, **w;
  double *err, *dy;

  double **A, *b, *b_hat, *c;
  int s, ord;

  int *first, *size;

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
  double **w, *y, *y0, *y_old, *err, *dy;
  double **A, *b, *b_hat, *c;
  double timer, err_max, h, t, tol, t0, te;
  int i, s, ord, first, last, size, me;
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
  w = shared->w;
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

  first = shared->first[me];
  size = shared->size[me];
  last = first + size - 1;

  printf("%d: %d %d %d\n", me, first, last, size);

  y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y + first, y0 + first, size);

  barrier_wait(bar);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    printf("%f %f %e %e\n", t0, te, t, h);

    err_max = 0.0;

    block_first_stage(first, size, s, t, h, A, b, b_hat, c, y, err, dy, w);

    for (i = 1; i < s - 1; i++)
    {
      barrier_wait(bar);
      block_interm_stage(i, first, size, s, t, h, A, b, b_hat, c, y, err, dy, w);
    }

    barrier_wait(bar);
    block_last_stage(first, size, s, t, h, b, b_hat, c, y, err, dy, w, &err_max);

    printf("e[%d]: %e\n", me, err_max);
    err_max = reduction_max(red, err_max);
    printf("e: %e\n", err_max);

    /* step control */

    step_control(&t, &h, err_max, ord, tol, y + first, y_old + first, size,
                 &steps_acc, &steps_rej);

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
  printf("Implementation variant: D (temporal locality of reads)\n");
  printf("Number of threads: %d\n", THREADS);

  arg = MALLOC(THREADS, arg_t);
  shared = MALLOC(1, shared_arg_t);
  arglist = MALLOC(THREADS, void *);

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

  ALLOC2D(shared->w, s, ode_size, double);
  shared->err = MALLOC(ode_size, double);
  shared->dy = MALLOC(ode_size, double);

  barrier_init(&shared->barrier, THREADS);
  reduction_init(&shared->reduction, THREADS);

  shared->first = MALLOC(THREADS, int);
  shared->size = MALLOC(THREADS, int);

  blockwise_distribution(THREADS, ode_size, shared->first, shared->size);

  for (i = 0; i < THREADS; i++)
  {
    arg[i].me = i;
    arg[i].shared = shared;
    arglist[i] = (void *) (arg + i);
  }

  run_threads(THREADS, solver_thread, arglist);

  FREE(shared->first);
  FREE(shared->size);

  barrier_destroy(&shared->barrier);
  reduction_destroy(&shared->reduction);

  free_emb_rk_method(&A, &b, &b_hat, &c, s);
  FREE(shared->b_hat);

  FREE2D(shared->w);
  FREE(shared->err);
  FREE(shared->dy);

  FREE(shared);
  FREE(arg);
  FREE(arglist);
}

/******************************************************************************/
