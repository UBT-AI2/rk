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

  int *block_offset, *block_length;

  barrier_t barrier;
  reduction_t reduction;

  mutex_lock_t **mutex_first;
  mutex_lock_t **mutex_last;

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
  int i, j;
  double **w, *y, *y0, *y_old, *err, *dy;
  double **A, *b, *b_hat, *c;
  double timer, err_max, h, t, tol, t0, te;
  int s, ord, first_elem, last_elem, num_elems, me;
  int steps_acc = 0, steps_rej = 0;
  barrier_t *bar;
  reduction_t *red;
  shared_arg_t *shared;
  mutex_lock_t **mutex_first, **mutex_last;
  int me_is_even;

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

  mutex_first = shared->mutex_first;
  mutex_last = shared->mutex_last;

  first_elem = shared->block_offset[me] * BLOCKSIZE;
  num_elems = shared->block_length[me] * BLOCKSIZE;
  last_elem = first_elem + num_elems - 1;

  me_is_even = (me % 2 == 0);

  assert(s >= 2);               /* !!! at least two stages !!! */
  assert(num_elems >= 2 * s * BLOCKSIZE);       /* !!! at least 2s blocks per thread !!! */

  y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y + first_elem, y0 + first_elem, num_elems);

  barrier_wait(bar);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    err_max = 0.0;

    init_mutexes(me, s, mutex_first, mutex_last);
    barrier_wait(bar);

    /* initialize the pipeline */

    for (j = 1; j < s; j++)
    {
      block_scatter_first_stage(first_elem + (2 * j - 1) * BLOCKSIZE, BLOCKSIZE,
                                s, t, h, A, b, b_hat, c, y, err, dy, w);
      for (i = 1; i < j; i++)
        block_scatter_interm_stage(i, first_elem + (2 * j - 1 - i) * BLOCKSIZE,
                                   BLOCKSIZE, s, t, h, A, b, b_hat, c, y, err,
                                   dy, w);

      block_scatter_first_stage(first_elem + 2 * j * BLOCKSIZE, BLOCKSIZE, s, t,
                                h, A, b, b_hat, c, y, err, dy, w);
      for (i = 1; i < j; i++)
        block_scatter_interm_stage(i, first_elem + (2 * j - i) * BLOCKSIZE,
                                   BLOCKSIZE, s, t, h, A, b, b_hat, c, y, err,
                                   dy, w);
    }

    /* sweep */

    for (j = first_elem + (2 * s - 1) * BLOCKSIZE;
         j < last_elem - BLOCKSIZE + 1; j += BLOCKSIZE)
    {
      block_scatter_first_stage(j, BLOCKSIZE, s, t, h, A, b, b_hat, c, y, err,
                                dy, w);

      for (i = 1; i < s - 1; i++)
        block_scatter_interm_stage(i, j - i * BLOCKSIZE, BLOCKSIZE, s, t, h, A,
                                   b, b_hat, c, y, err, dy, w);

      block_scatter_last_stage(j - ((s - 1) * BLOCKSIZE), BLOCKSIZE, s, t, h, b,
                               b_hat, c, y, err, dy, w, &err_max);
    }

    /* finalization */

    if (me_is_even)
      goto finalize_high;

  finalize_low:

    block_scatter_first_stage(first_elem, BLOCKSIZE, s, t, h, A, b, b_hat, c, y,
                              err, dy, w);
    first_block_complete(me, 1, mutex_first);

    for (i = 1; i < s - 1; i++)
      block_scatter_interm_stage(i, first_elem + i * BLOCKSIZE, BLOCKSIZE, s, t,
                                 h, A, b, b_hat, c, y, err, dy, w);

    block_scatter_last_stage(first_elem + (s - 1) * BLOCKSIZE, BLOCKSIZE, s, t,
                             h, b, b_hat, c, y, err, dy, w, &err_max);

    for (j = 1; j < s - 1; j++)
    {
      wait_for_pred(me, j, mutex_last);
      block_scatter_interm_stage(j, first_elem, BLOCKSIZE, s, t, h, A, b, b_hat,
                                 c, y, err, dy, w);
      release_pred(me, j, mutex_last);
      first_block_complete(me, j + 1, mutex_first);

      for (i = j + 1; i < s - 1; i++)
        block_scatter_interm_stage(i, first_elem + (i - j) * BLOCKSIZE,
                                   BLOCKSIZE, s, t, h, A, b, b_hat, c, y, err,
                                   dy, w);

      block_scatter_last_stage(first_elem + (s - 1 - j) * BLOCKSIZE, BLOCKSIZE,
                               s, t, h, b, b_hat, c, y, err, dy, w, &err_max);
    }

    wait_for_pred(me, s - 1, mutex_last);
    block_scatter_last_stage(first_elem, BLOCKSIZE, s, t, h, b, b_hat, c, y,
                             err, dy, w, &err_max);
    release_pred(me, s - 1, mutex_last);

    if (me_is_even)
      goto step_control;

  finalize_high:

    block_scatter_first_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t, h, A,
                              b, b_hat, c, y, err, dy, w);
    last_block_complete(me, 1, mutex_last);

    for (i = 1; i < s - 1; i++)
      block_scatter_interm_stage(i, last_elem - BLOCKSIZE + 1 - i * BLOCKSIZE,
                                 BLOCKSIZE, s, t, h, A, b, b_hat, c, y, err, dy,
                                 w);

    block_scatter_last_stage(last_elem - BLOCKSIZE + 1 - (s - 1) * BLOCKSIZE,
                             BLOCKSIZE, s, t, h, b, b_hat, c, y, err, dy, w,
                             &err_max);

    for (j = 1; j < s - 1; j++)
    {
      wait_for_succ(me, j, mutex_first);
      block_scatter_interm_stage(j, last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t,
                                 h, A, b, b_hat, c, y, err, dy, w);
      release_succ(me, j, mutex_first);
      last_block_complete(me, j + 1, mutex_last);

      for (i = j + 1; i < s - 1; i++)
        block_scatter_interm_stage(i,
                                   last_elem - BLOCKSIZE + 1 - (i -
                                                                j) * BLOCKSIZE,
                                   BLOCKSIZE, s, t, h, A, b, b_hat, c, y, err,
                                   dy, w);

      block_scatter_last_stage(last_elem - BLOCKSIZE + 1 -
                               (s - 1 - j) * BLOCKSIZE, BLOCKSIZE, s, t, h, b,
                               b_hat, c, y, err, dy, w, &err_max);
    }

    wait_for_succ(me, s - 1, mutex_first);
    block_scatter_last_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t, h, b,
                             b_hat, c, y, err, dy, w, &err_max);
    release_succ(me, s - 1, mutex_first);

    if (me_is_even)
      goto finalize_low;

  step_control:

    err_max = reduction_max(red, err_max);

    /* step control */

    step_control(&t, &h, err_max, ord, tol, y + first_elem, y_old + first_elem,
                 num_elems, &steps_acc, &steps_rej);
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
  int i, j, s, ord;

  printf("Solver type: ");
  printf("parallel embedded Runge-Kutta method for shared address space\n");
  printf("Implementation variant: PipeD ");
  printf("(pipelining scheme based on implementation D)\n");
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

  ALLOC2D(shared->w, s, ode_size, double);
  shared->err = MALLOC(ode_size, double);
  shared->dy = MALLOC(ode_size, double);

  barrier_init(&shared->barrier, threads);
  reduction_init(&shared->reduction, threads);

  ALLOC2D(shared->mutex_first, threads, s, mutex_lock_t);
  ALLOC2D(shared->mutex_last, threads, s, mutex_lock_t);

  for (i = 0; i < threads; i++)
    for (j = 0; j < s; j++)
    {
      mutex_lock_init(&(shared->mutex_first[i][j]));
      mutex_lock_init(&(shared->mutex_last[i][j]));
    }

  shared->block_offset = MALLOC(threads, int);
  shared->block_length = MALLOC(threads, int);

  assert(ode_size % BLOCKSIZE == 0);
  blockwise_distribution(threads, ode_size / BLOCKSIZE, shared->block_offset,
                         shared->block_length);

  for (i = 0; i < threads; i++)
  {
    arg[i].me = i;
    arg[i].shared = shared;
    arglist[i] = (void *) (arg + i);
  }

  run_threads(threads, solver_thread, arglist);

  for (i = 0; i < threads; i++)
    for (j = 0; j < s; j++)
    {
      mutex_lock_destroy(&(shared->mutex_first[i][j]));
      mutex_lock_destroy(&(shared->mutex_last[i][j]));
    }

  FREE2D(shared->mutex_first);
  FREE2D(shared->mutex_last);

  FREE(shared->block_offset);
  FREE(shared->block_length);

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
