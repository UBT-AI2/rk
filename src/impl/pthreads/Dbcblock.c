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
  double **w, *y, *y0, *y_old, *err, *dy, *v;
  double **A, *b, *b_hat, *c;
  double timer, err_max, h, t, tol, t0, te;
  int i, s, ord, first_elem, last_elem, num_elems, me;
  int steps_acc = 0, steps_rej = 0;
  barrier_t *bar;
  reduction_t *red;
  shared_arg_t *shared;
  mutex_lock_t **mutex_first, **mutex_last;

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

  y_old = dy;

  v = MALLOC(BLOCKSIZE, double);

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y + first_elem, y0 + first_elem, num_elems);

  barrier_wait(bar);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    err_max = 0.0;

    init_mutexes(me, s, mutex_first, mutex_last);
    barrier_wait(bar);

    /* evaluate the inner blocks of the first stage */

    tiled_block_scatter_first_stage(first_elem + BLOCKSIZE,
                                    num_elems - 2 * BLOCKSIZE, s, t, h, A, b,
                                    b_hat, c, y, err, dy, w, v);

    /* evaluate first block of the first stage and send result to the
       previous processor */

    tiled_block_scatter_first_stage(first_elem, BLOCKSIZE, s, t, h, A, b, b_hat,
                                    c, y, err, dy, w, v);
    first_block_complete(me, 1, mutex_first);

    /* evaluate last block of the second stage and send result to the
       next processor */

    tiled_block_scatter_first_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t,
                                    h, A, b, b_hat, c, y, err, dy, w, v);
    last_block_complete(me, 1, mutex_last);

    for (i = 1; i < s - 1; i++)
    {
      /* evaluate the inner blocks of stage i */

      tiled_block_scatter_interm_stage(i, first_elem + BLOCKSIZE,
                                       num_elems - 2 * BLOCKSIZE, s, t, h, A, b,
                                       b_hat, c, y, err, dy, w, v);

      /* evaluate first block of stage i and send result to the
         previous processor */

      wait_for_pred(me, i, mutex_last);
      tiled_block_scatter_interm_stage(i, first_elem, BLOCKSIZE, s, t, h, A, b,
                                       b_hat, c, y, err, dy, w, v);
      first_block_complete(me, i + 1, mutex_first);
      release_pred(me, i, mutex_last);

      /* evaluate last block of stage i and send result to the next
         processor */

      wait_for_succ(me, i, mutex_first);
      tiled_block_scatter_interm_stage(i, last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                       s, t, h, A, b, b_hat, c, y, err, dy, w,
                                       v);
      last_block_complete(me, i + 1, mutex_last);
      release_succ(me, i, mutex_first);
    }

    /* evaluate the inner blocks of stage s - 1 */

    tiled_block_scatter_last_stage(first_elem + BLOCKSIZE,
                                   num_elems - 2 * BLOCKSIZE, s, t, h, b, b_hat,
                                   c, y, err, dy, w, v, &err_max);

    /* evaluate first block of stage s - 1 */

    wait_for_pred(me, s - 1, mutex_last);
    tiled_block_scatter_last_stage(first_elem, BLOCKSIZE, s, t, h, b, b_hat, c,
                                   y, err, dy, w, v, &err_max);
    release_pred(me, s - 1, mutex_last);

    /* evaluate last block of stage s - 1 */

    wait_for_succ(me, s - 1, mutex_first);
    tiled_block_scatter_last_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t,
                                   h, b, b_hat, c, y, err, dy, w, v, &err_max);
    release_succ(me, s - 1, mutex_first);

    err_max = reduction_max(red, err_max);

    /* step control */

    step_control(&t, &h, err_max, ord, tol, y + first_elem, y_old + first_elem,
                 num_elems, &steps_acc, &steps_rej);
  }

  timer_stop(&timer);

  if (me == 0)
    print_statistics(timer, steps_acc, steps_rej);

  FREE(v);

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
  printf("Implementation variant: Dbc (D with block-based communication)\n");
  printf("communication)\n");
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
