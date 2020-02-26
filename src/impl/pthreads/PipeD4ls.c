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

  double *y, *y0, ***w;

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
  int i, j, k;
  double **w, *y, *y0, *y_old, *err, *dy;
  double *buf;
  double **A, *b, *b_hat, *c;
  double timer, err_max, h, t, tol, t0, te;
  int s, ord, first_elem, last_elem, num_elems, me;
  int steps_acc = 0, steps_rej = 0;
  barrier_t *bar;
  reduction_t *red;
  shared_arg_t *shared;
  mutex_lock_t **mutex_first, **mutex_last;
  double **w_pred, **w_succ;
  int me_is_even, me_is_first_thread, me_is_last_thread;
  int num_blocks;

  me = ((arg_t *) argument)->me;
  shared = ((arg_t *) argument)->shared;

  t0 = shared->t0;
  te = shared->te;
  tol = shared->tol;

  y0 = shared->y0;
  y = shared->y;

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

  num_blocks = shared->block_length[me];

  first_elem = shared->block_offset[me] * BLOCKSIZE;
  num_elems = shared->block_length[me] * BLOCKSIZE;
  last_elem = first_elem + num_elems - 1;

  assert(s >= 2);               /* !!! at least two stages !!! */
  assert(num_blocks >= 2);      /* !!! at least 2 blocks per thread !!! */

  me_is_first_thread = (me == 0);
  me_is_last_thread = (me == threads - 1);

  me_is_even = (me % 2 == 0);

  buf =
    MALLOC(BLOCKSIZE + last_elem - first_elem + 1 +
           (s * s + 5 * s - 4) * BLOCKSIZE / 2, double);
  dy = buf - first_elem;
  w = MALLOC(s, double *);

  if (me_is_even)
  {
    err = dy + s * BLOCKSIZE;
    shared->w[me][1] = w[1] = err + 3 * BLOCKSIZE;

    for (i = 2; i < s; ++i)
      shared->w[me][i] = w[i] = w[i - 1] + (i + 2) * BLOCKSIZE;
  }
  else
  {
    dy += (s * s + 5 * s - 4) * BLOCKSIZE / 2 + BLOCKSIZE;
    err = dy - s * BLOCKSIZE;
    shared->w[me][1] = w[1] = err - 3 * BLOCKSIZE;

    for (i = 2; i < s; ++i)
      shared->w[me][i] = w[i] = w[i - 1] - (i + 2) * BLOCKSIZE;
  }

  w_pred = me_is_first_thread ? NULL : shared->w[me - 1];
  w_succ = me_is_last_thread ? NULL : shared->w[me + 1];

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

    if (num_blocks < s + 1)     /* at most s blocks per thread */
    {
      if (me_is_even)
      {
        lock_init_phase(me, mutex_first);

        /* triangle (0, 0) -- (num_blocks-2, 0) -- (0, num_blocks-2) */

        block_scatter_first_stage(first_elem, BLOCKSIZE, s, t, h, A, b, b_hat,
                                  c, y, err, dy, w);
        first_block_complete(me, 1, mutex_first);

        for (j = first_elem + BLOCKSIZE; j < last_elem - BLOCKSIZE + 1;
             j += BLOCKSIZE)
        {
          block_scatter_first_stage(j, BLOCKSIZE, s, t, h, A, b, b_hat, c, y,
                                    err, dy, w);

          for (i = j - BLOCKSIZE, k = 1; i > first_elem; i -= BLOCKSIZE)
            block_scatter_interm_stage(k++, i, BLOCKSIZE, s, t, h, A, b, b_hat,
                                       c, y, err, dy, w);

          get_from_pred(me, k, w, w_pred, first_elem, BLOCKSIZE, mutex_last);
          block_scatter_interm_stage(k, first_elem, BLOCKSIZE, s, t, h, A, b,
                                     b_hat, c, y, err, dy, w);
          first_block_complete(me, ++k, mutex_first);
        }

        /* parallelogram (num_blocks-1, 0) -- (0, num_blocks-1) --
           (0, s-1) -- (num_blocks-1, s-num_blocks) */

        block_scatter_first_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t, h,
                                  A, b, b_hat, c, y, err, dy, w);
        last_block_complete(me, 1, mutex_last);

        for (i = last_elem - 2 * BLOCKSIZE + 1, k = 1; i > first_elem;
             i -= BLOCKSIZE)
          block_scatter_interm_stage(k++, i, BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                     y, err, dy, w);

        if (num_blocks < s)
        {
          get_from_pred(me, k, w, w_pred, first_elem, BLOCKSIZE, mutex_last);

          block_scatter_interm_stage(num_blocks - 1, first_elem, BLOCKSIZE, s,
                                     t, h, A, b, b_hat, c, y, err, dy, w);
          first_block_complete(me, num_blocks, mutex_first);

          for (j = 1; j < s - num_blocks; j++)
          {
            k = j + 1;

            get_from_succ(me, j, w, w_succ, last_elem, BLOCKSIZE, mutex_first);
            block_scatter_interm_stage(j, last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                       s, t, h, A, b, b_hat, c, y, err, dy, w);
            last_block_complete(me, k, mutex_last);

            for (i = last_elem - 2 * BLOCKSIZE + 1; i > first_elem;
                 i -= BLOCKSIZE)
              block_scatter_interm_stage(k++, i, BLOCKSIZE, s, t, h, A, b,
                                         b_hat, c, y, err, dy, w);

            get_from_pred(me, k, w, w_pred, first_elem, BLOCKSIZE, mutex_last);
            block_scatter_interm_stage(k, first_elem, BLOCKSIZE, s, t, h, A, b,
                                       b_hat, c, y, err, dy, w);
            first_block_complete(me, ++k, mutex_first);
          }

          k = s - num_blocks;

          get_from_succ(me, k, w, w_succ, last_elem, BLOCKSIZE, mutex_first);
          block_scatter_interm_stage(k, last_elem - BLOCKSIZE + 1, BLOCKSIZE, s,
                                     t, h, A, b, b_hat, c, y, err, dy, w);
          last_block_complete(me, ++k, mutex_last);

          for (i = last_elem - 2 * BLOCKSIZE + 1; i > first_elem;
               i -= BLOCKSIZE)
            block_scatter_interm_stage(k++, i, BLOCKSIZE, s, t, h, A, b, b_hat,
                                       c, y, err, dy, w);
        }

        get_from_pred(me, s - 1, w, w_pred, first_elem, BLOCKSIZE, mutex_last);
        unlock_init_phase(me, mutex_first);
        block_scatter_last_stage(first_elem, BLOCKSIZE, s, t, h, b, b_hat, c, y,
                                 err, dy, w, &err_max);

        wait_pred_init_complete(me, mutex_last);

        /* triangle (1, s-1) -- (num_blocks-1, s-num_blocks+1) -- (s-1, s-1) */

        for (j = s - num_blocks + 1; j < s - 1; j++)
        {
          k = j + 1;

          get_from_succ(me, j, w, w_succ, last_elem, BLOCKSIZE, mutex_first);
          block_scatter_interm_stage(j, last_elem - BLOCKSIZE + 1, BLOCKSIZE, s,
                                     t, h, A, b, b_hat, c, y, err, dy, w);
          last_block_complete(me, k, mutex_last);

          for (i = last_elem - 2 * BLOCKSIZE + 1; k < s - 1; i -= BLOCKSIZE)
            block_scatter_interm_stage(k++, i, BLOCKSIZE, s, t, h, A, b, b_hat,
                                       c, y, err, dy, w);

          block_scatter_last_stage(i, BLOCKSIZE, s, t, h, b, b_hat, c, y, err,
                                   dy, w, &err_max);
        }

        get_from_succ(me, s - 1, w, w_succ, last_elem, BLOCKSIZE, mutex_first);

        block_scatter_last_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t, h,
                                 b, b_hat, c, y, err, dy, w, &err_max);
      }
      else                      /* !me_is_even */
      {
        lock_init_phase(me, mutex_last);

        /* triangle (0, 1) -- (num_blocks-1, 0) --
           (num_blocks-1, num_blocks-2) */

        block_scatter_first_stage_reverse(last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                          s, t, h, A, b, b_hat, c, y, err, dy,
                                          w);
        last_block_complete(me, 1, mutex_last);

        for (j = last_elem - 2 * BLOCKSIZE + 1; j > first_elem; j -= BLOCKSIZE)
        {
          block_scatter_first_stage_reverse(j, BLOCKSIZE, s, t, h, A, b, b_hat,
                                            c, y, err, dy, w);

          for (i = j + BLOCKSIZE, k = 1; i < last_elem - BLOCKSIZE + 1;
               i += BLOCKSIZE)
            block_scatter_interm_stage_reverse(k++, i, BLOCKSIZE, s, t, h, A, b,
                                               b_hat, c, y, err, dy, w);

          get_from_succ(me, k, w, w_succ, last_elem, BLOCKSIZE, mutex_first);
          block_scatter_interm_stage_reverse(k, last_elem - BLOCKSIZE + 1,
                                             BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                             y, err, dy, w);
          last_block_complete(me, ++k, mutex_last);
        }

        /* parallelogram (0, 0) -- (num_blocks-1, num_blocks-1) --
           (num_block-1, s-1) -- (0, s-num_blocks) */

        block_scatter_first_stage_reverse(first_elem, BLOCKSIZE, s, t, h, A, b,
                                          b_hat, c, y, err, dy, w);
        first_block_complete(me, 1, mutex_first);

        for (i = first_elem + BLOCKSIZE, k = 1; i < last_elem - BLOCKSIZE + 1;
             i += BLOCKSIZE)
          block_scatter_interm_stage_reverse(k++, i, BLOCKSIZE, s, t, h, A, b,
                                             b_hat, c, y, err, dy, w);

        if (num_blocks < s)     /* less than s blocks per thread */
        {
          get_from_succ(me, k, w, w_succ, last_elem, BLOCKSIZE, mutex_first);

          block_scatter_interm_stage_reverse(num_blocks - 1,
                                             last_elem - BLOCKSIZE + 1,
                                             BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                             y, err, dy, w);
          last_block_complete(me, num_blocks, mutex_last);

          for (j = 1; j < s - num_blocks; j++)
          {
            k = j + 1;

            get_from_pred(me, j, w, w_pred, first_elem, BLOCKSIZE, mutex_last);
            block_scatter_interm_stage_reverse(j, first_elem, BLOCKSIZE, s, t,
                                               h, A, b, b_hat, c, y, err, dy,
                                               w);
            first_block_complete(me, k, mutex_first);

            for (i = first_elem + BLOCKSIZE; i < last_elem - BLOCKSIZE + 1;
                 i += BLOCKSIZE)
              block_scatter_interm_stage_reverse(k++, i, BLOCKSIZE, s, t, h, A,
                                                 b, b_hat, c, y, err, dy, w);

            get_from_succ(me, k, w, w_succ, last_elem, BLOCKSIZE, mutex_first);
            block_scatter_interm_stage(k, last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                       s, t, h, A, b, b_hat, c, y, err, dy, w);
            last_block_complete(me, ++k, mutex_last);
          }

          k = s - num_blocks;

          get_from_pred(me, k, w, w_pred, first_elem, BLOCKSIZE, mutex_last);
          block_scatter_interm_stage_reverse(k, first_elem, BLOCKSIZE, s, t, h,
                                             A, b, b_hat, c, y, err, dy, w);
          first_block_complete(me, ++k, mutex_first);

          for (i = first_elem + BLOCKSIZE; i < last_elem - BLOCKSIZE + 1;
               i += BLOCKSIZE)
            block_scatter_interm_stage_reverse(k++, i, BLOCKSIZE, s, t, h, A, b,
                                               b_hat, c, y, err, dy, w);
        }

        get_from_succ(me, s - 1, w, w_succ, last_elem, BLOCKSIZE, mutex_first);
        unlock_init_phase(me, mutex_last);
        block_scatter_last_stage_reverse(last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                         s, t, h, b, b_hat, c, y, err, dy, w,
                                         &err_max);

        wait_succ_init_complete(me, mutex_first);

        /* triangle (num_blocks-2, s-1) -- (0, s-num_blocks+1) -- (0, s-1) */

        for (j = s - num_blocks + 1; j < s - 1; j++)
        {
          k = j + 1;

          get_from_pred(me, j, w, w_pred, first_elem, BLOCKSIZE, mutex_last);
          block_scatter_interm_stage_reverse(j, first_elem, BLOCKSIZE, s, t, h,
                                             A, b, b_hat, c, y, err, dy, w);
          first_block_complete(me, k, mutex_first);

          for (i = first_elem + BLOCKSIZE; k < s - 1; i += BLOCKSIZE)
            block_scatter_interm_stage_reverse(k++, i, BLOCKSIZE, s, t, h, A, b,
                                               b_hat, c, y, err, dy, w);

          block_scatter_last_stage_reverse(i, BLOCKSIZE, s, t, h, b, b_hat, c,
                                           y, err, dy, w, &err_max);
        }

        get_from_pred(me, s - 1, w, w_pred, first_elem, BLOCKSIZE, mutex_last);

        block_scatter_last_stage_reverse(first_elem, BLOCKSIZE, s, t, h, b,
                                         b_hat, c, y, err, dy, w, &err_max);
      }
    }
    else                        /* more than s blocks per thread  */
    {
      if (me_is_even)
      {
        /* initialize the pipeline */

        lock_init_phase(me, mutex_first);

        block_scatter_first_stage(first_elem, BLOCKSIZE, s, t, h, A, b, b_hat,
                                  c, y, err, dy, w);
        first_block_complete(me, 1, mutex_first);

        for (j = 1; j < s - 1; j++)
        {
          block_scatter_first_stage(first_elem + j * BLOCKSIZE, BLOCKSIZE, s, t,
                                    h, A, b, b_hat, c, y, err, dy, w);
          for (i = 1; i < j; i++)
            block_scatter_interm_stage(i,
                                       first_elem + j * BLOCKSIZE -
                                       i * BLOCKSIZE, BLOCKSIZE, s, t, h, A, b,
                                       b_hat, c, y, err, dy, w);
          get_from_pred(me, j, w, w_pred, first_elem, BLOCKSIZE, mutex_last);
          block_scatter_interm_stage(j, first_elem, BLOCKSIZE, s, t, h, A, b,
                                     b_hat, c, y, err, dy, w);
          first_block_complete(me, j + 1, mutex_first);
        }

        block_scatter_first_stage(first_elem + (s - 1) * BLOCKSIZE, BLOCKSIZE,
                                  s, t, h, A, b, b_hat, c, y, err, dy, w);
        for (i = 1; i < j; i++)
          block_scatter_interm_stage(i,
                                     first_elem + (s - 1) * BLOCKSIZE -
                                     i * BLOCKSIZE, BLOCKSIZE, s, t, h, A, b,
                                     b_hat, c, y, err, dy, w);
        get_from_pred(me, s - 1, w, w_pred, first_elem, BLOCKSIZE, mutex_last);

        unlock_init_phase(me, mutex_first);

        block_scatter_last_stage(first_elem, BLOCKSIZE, s, t, h, b, b_hat, c, y,
                                 err, dy, w, &err_max);

        wait_pred_init_complete(me, mutex_last);

        /* sweep */

        for (j = first_elem + s * BLOCKSIZE; j < last_elem - BLOCKSIZE + 1;
             j += BLOCKSIZE)
        {
          block_scatter_first_stage(j, BLOCKSIZE, s, t, h, A, b, b_hat, c, y,
                                    err, dy, w);
          for (i = 1; i < s - 1; i++)
            block_scatter_interm_stage(i, j - i * BLOCKSIZE, BLOCKSIZE, s, t, h,
                                       A, b, b_hat, c, y, err, dy, w);
          block_scatter_last_stage(j - (s - 1) * BLOCKSIZE, BLOCKSIZE, s, t, h,
                                   b, b_hat, c, y, err, dy, w, &err_max);
        }

        /* finalization */

        block_scatter_first_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t, h,
                                  A, b, b_hat, c, y, err, dy, w);
        last_block_complete(me, 1, mutex_last);

        for (i = 1; i < s - 1; i++)
          block_scatter_interm_stage(i, last_elem - (i + 1) * BLOCKSIZE + 1,
                                     BLOCKSIZE, s, t, h, A, b, b_hat, c, y, err,
                                     dy, w);
        block_scatter_last_stage(last_elem - s * BLOCKSIZE + 1, BLOCKSIZE, s, t,
                                 h, b, b_hat, c, y, err, dy, w, &err_max);

        for (j = 1; j < s - 1; j++)
        {
          get_from_succ(me, j, w, w_succ, last_elem, BLOCKSIZE, mutex_first);
          block_scatter_interm_stage(j, last_elem - BLOCKSIZE + 1, BLOCKSIZE, s,
                                     t, h, A, b, b_hat, c, y, err, dy, w);
          last_block_complete(me, j + 1, mutex_last);

          for (i = j + 1; i < s - 1; i++)
            block_scatter_interm_stage(i,
                                       last_elem - (i - j + 1) * BLOCKSIZE + 1,
                                       BLOCKSIZE, s, t, h, A, b, b_hat, c, y,
                                       err, dy, w);

          block_scatter_last_stage(last_elem - (s - j) * BLOCKSIZE + 1,
                                   BLOCKSIZE, s, t, h, b, b_hat, c, y, err, dy,
                                   w, &err_max);
        }

        get_from_succ(me, s - 1, w, w_succ, last_elem, BLOCKSIZE, mutex_first);

        block_scatter_last_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t, h,
                                 b, b_hat, c, y, err, dy, w, &err_max);
      }
      else                      /* !me_is_even */
      {
        /* initialize the pipeline */

        lock_init_phase(me, mutex_last);

        block_scatter_first_stage_reverse(last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                          s, t, h, A, b, b_hat, c, y, err, dy,
                                          w);
        last_block_complete(me, 1, mutex_last);

        for (j = 1; j < s - 1; j++)
        {
          block_scatter_first_stage_reverse(last_elem - (j + 1) * BLOCKSIZE + 1,
                                            BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                            y, err, dy, w);
          for (i = 1; i < j; i++)
            block_scatter_interm_stage_reverse(i,
                                               last_elem - (j - i +
                                                            1) * BLOCKSIZE + 1,
                                               BLOCKSIZE, s, t, h, A, b, b_hat,
                                               c, y, err, dy, w);
          get_from_succ(me, j, w, w_succ, last_elem, BLOCKSIZE, mutex_first);
          block_scatter_interm_stage_reverse(j, last_elem - BLOCKSIZE + 1,
                                             BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                             y, err, dy, w);
          last_block_complete(me, j + 1, mutex_last);
        }

        block_scatter_first_stage_reverse(last_elem - s * BLOCKSIZE + 1,
                                          BLOCKSIZE, s, t, h, A, b, b_hat, c, y,
                                          err, dy, w);
        for (i = 1; i < j; i++)
          block_scatter_interm_stage_reverse(i,
                                             last_elem - (s - i) * BLOCKSIZE +
                                             1, BLOCKSIZE, s, t, h, A, b, b_hat,
                                             c, y, err, dy, w);
        get_from_succ(me, s - 1, w, w_succ, last_elem, BLOCKSIZE, mutex_first);

        unlock_init_phase(me, mutex_last);

        block_scatter_last_stage_reverse(last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                         s, t, h, b, b_hat, c, y, err, dy, w,
                                         &err_max);

        wait_succ_init_complete(me, mutex_first);

        /* sweep */

        for (j = last_elem - (s + 1) * BLOCKSIZE + 1; j > first_elem;
             j -= BLOCKSIZE)
        {
          block_scatter_first_stage_reverse(j, BLOCKSIZE, s, t, h, A, b, b_hat,
                                            c, y, err, dy, w);
          for (i = 1; i < s - 1; i++)
            block_scatter_interm_stage_reverse(i, j + i * BLOCKSIZE, BLOCKSIZE,
                                               s, t, h, A, b, b_hat, c, y, err,
                                               dy, w);
          block_scatter_last_stage_reverse(j + (s - 1) * BLOCKSIZE, BLOCKSIZE,
                                           s, t, h, b, b_hat, c, y, err, dy, w,
                                           &err_max);
        }

        /* finalization */

        block_scatter_first_stage_reverse(first_elem, BLOCKSIZE, s, t, h, A, b,
                                          b_hat, c, y, err, dy, w);
        first_block_complete(me, 1, mutex_first);

        for (i = 1; i < s - 1; i++)
          block_scatter_interm_stage_reverse(i, first_elem + i * BLOCKSIZE,
                                             BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                             y, err, dy, w);

        block_scatter_last_stage_reverse(first_elem + (s - 1) * BLOCKSIZE,
                                         BLOCKSIZE, s, t, h, b, b_hat, c, y,
                                         err, dy, w, &err_max);

        for (j = 1; j < s - 1; j++)
        {
          get_from_pred(me, j, w, w_pred, first_elem, BLOCKSIZE, mutex_last);
          block_scatter_interm_stage_reverse(j, first_elem, BLOCKSIZE, s, t, h,
                                             A, b, b_hat, c, y, err, dy, w);
          first_block_complete(me, j + 1, mutex_first);

          for (i = j + 1; i < s - 1; i++)
            block_scatter_interm_stage_reverse(i,
                                               first_elem + (i - j) * BLOCKSIZE,
                                               BLOCKSIZE, s, t, h, A, b, b_hat,
                                               c, y, err, dy, w);

          block_scatter_last_stage_reverse(first_elem + (s - 1 - j) * BLOCKSIZE,
                                           BLOCKSIZE, s, t, h, b, b_hat, c, y,
                                           err, dy, w, &err_max);
        }

        get_from_pred(me, s - 1, w, w_pred, first_elem, BLOCKSIZE, mutex_last);
        block_scatter_last_stage_reverse(first_elem, BLOCKSIZE, s, t, h, b,
                                         b_hat, c, y, err, dy, w, &err_max);
      }
    }

    /* step control */

    err_max = reduction_max(red, err_max);

    step_control(&t, &h, err_max, ord, tol, y + first_elem, y_old + first_elem,
                 num_elems, &steps_acc, &steps_rej);
  }

  timer_stop(&timer);

  if (me == 0)
    print_statistics(timer, steps_acc, steps_rej);

  FREE(w);
  FREE(buf);

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
  printf("Implementation variant: PipeDls ");
  printf("(low-storage pipelining scheme based on implementation D)\n");
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

  ALLOC2D(shared->w, threads, s, double *);

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

  FREE(shared);
  FREE(arg);
  FREE(arglist);
}

/******************************************************************************/
