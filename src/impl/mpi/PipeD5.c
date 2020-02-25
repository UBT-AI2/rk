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

void solver(double t0, double te, double *y0, double *y, double tol)
{
  int i, j, k;
  double **w, *y_old, *err, *dy;
  double **A, *b, *b_hat, *c;
  double err_max, my_err_max;
  int s, ord;
  double h, t;
  double timer;
  int steps_acc = 0, steps_rej = 0;
  int first_elem, num_elems, last_elem, *elem_offset, *elem_length;
  int num_blocks, *block_offset, *block_length;
  int first_elem_y, last_elem_y, num_elems_y;
  MPI_Request recv_req_succ, send_req_succ, recv_req_pred, send_req_pred;
  MPI_Status status;
  int me_is_even;

  if (me == 0)
  {
    printf("Solver type: parallel embedded Runge-Kutta method");
    printf(" for distributed address space\n");
    printf("Implementation variant: PipeD5 ");
    printf("(pipelining scheme based on implementation D ");
    printf("with alternative computation order)\n");
    printf("Number of MPI processes: %d\n", processes);
  }

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  for (i = 0; i < s; i++)
    b_hat[i] = b[i] - b_hat[i];

  ALLOC2D(w, s, ode_size, double);

  err = MALLOC(ode_size, double);
  dy = MALLOC(ode_size, double);

  elem_offset = MALLOC(processes, int);
  elem_length = MALLOC(processes, int);

  block_offset = MALLOC(processes, int);
  block_length = MALLOC(processes, int);

  assert(ode_size % BLOCKSIZE == 0);
  blockwise_distribution(processes, ode_size / BLOCKSIZE, block_offset,
                         block_length);

  num_blocks = block_length[me];

  for (i = 0; i < processes; i++)
  {
    elem_offset[i] = block_offset[i] * BLOCKSIZE;
    elem_length[i] = block_length[i] * BLOCKSIZE;
  }

  first_elem = elem_offset[me];
  num_elems = elem_length[me];
  last_elem = first_elem + num_elems - 1;

  me_is_even = (me % 2 == 0);

  first_elem_y = (me_is_even && me != 0) ? first_elem - BLOCKSIZE : first_elem;
  last_elem_y = (!me_is_even
                 && me != processes - 1) ? last_elem + BLOCKSIZE : last_elem;
  num_elems_y = last_elem_y - first_elem_y + 1;

  assert(s >= 2);               /* !!! at least two stages !!! */
  assert(num_blocks >= 2);      /* !!! at least 2 blocks per process !!! */

  y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y + first_elem_y, y0 + first_elem_y, num_elems_y);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    my_err_max = 0.0;

    if (num_blocks < s + 1)
    {
      if (me_is_even)           /* even processor */
      {
        /* send last block of y to next processor and
           start receive operation for the first block of the
           next processor */

        start_recv_succ(y, last_elem, BLOCKSIZE, 0, &recv_req_succ);
        start_send_succ(y, last_elem, BLOCKSIZE, 0, &send_req_succ);

        /* triangle (0, 0) -- (num_blocks-2, 0) -- (0, num_blocks-2) */

        block_scatter_first_stage(first_elem, BLOCKSIZE, s, t, h, A, b, b_hat,
                                  c, y, err, dy, w);
        start_recv_pred(w[1], first_elem, BLOCKSIZE, 1, &recv_req_pred);
        start_send_pred(w[1], first_elem, BLOCKSIZE, 1, &send_req_pred);

        for (j = first_elem + BLOCKSIZE; j < last_elem - BLOCKSIZE + 1;
             j += BLOCKSIZE)
        {
          block_scatter_first_stage(j, BLOCKSIZE, s, t, h, A, b, b_hat, c, y,
                                    err, dy, w);

          for (i = j - BLOCKSIZE, k = 1; i > first_elem; i -= BLOCKSIZE)
            block_scatter_interm_stage(k++, i, BLOCKSIZE, s, t, h, A, b, b_hat,
                                       c, y, err, dy, w);

          complete_recv_pred(&recv_req_pred, &status);

          block_scatter_interm_stage(k++, first_elem, BLOCKSIZE, s, t, h, A, b,
                                     b_hat, c, y, err, dy, w);

          start_recv_pred(w[k], first_elem, BLOCKSIZE, k, &recv_req_pred);
          complete_send_pred(&send_req_pred, &status);
          start_send_pred(w[k], first_elem, BLOCKSIZE, k, &send_req_pred);
        }

        /* parallelogram (num_blocks-1, 0) -- (0, num_blocks-1) --
           (0, s-1) -- (num_blocks-1, s-num_blocks) */

        complete_recv_succ(&recv_req_succ, &status);

        block_scatter_first_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t, h,
                                  A, b, b_hat, c, y, err, dy, w);
        start_recv_succ(w[1], last_elem, BLOCKSIZE, 1, &recv_req_succ);
        complete_send_succ(&send_req_succ, &status);
        start_send_succ(w[1], last_elem, BLOCKSIZE, 1, &send_req_succ);

        for (i = last_elem - 2 * BLOCKSIZE + 1, k = 1; i > first_elem;
             i -= BLOCKSIZE)
          block_scatter_interm_stage(k++, i, BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                     y, err, dy, w);

        if (num_blocks < s)
        {
          complete_recv_pred(&recv_req_pred, &status);

          block_scatter_interm_stage(num_blocks - 1, first_elem, BLOCKSIZE, s,
                                     t, h, A, b, b_hat, c, y, err, dy, w);

          start_recv_pred(w[num_blocks], first_elem, BLOCKSIZE, num_blocks,
                          &recv_req_pred);
          complete_send_pred(&send_req_pred, &status);
          start_send_pred(w[num_blocks], first_elem, BLOCKSIZE, num_blocks,
                          &send_req_pred);

          for (j = 1; j < s - num_blocks; j++)
          {
            k = j + 1;

            complete_recv_succ(&recv_req_succ, &status);
            block_scatter_interm_stage(j, last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                       s, t, h, A, b, b_hat, c, y, err, dy, w);
            start_recv_succ(w[k], last_elem, BLOCKSIZE, k, &recv_req_succ);
            complete_send_succ(&send_req_succ, &status);
            start_send_succ(w[k], last_elem, BLOCKSIZE, k, &send_req_succ);

            for (i = last_elem - 2 * BLOCKSIZE + 1; i > first_elem;
                 i -= BLOCKSIZE)
              block_scatter_interm_stage(k++, i, BLOCKSIZE, s, t, h, A, b,
                                         b_hat, c, y, err, dy, w);

            complete_recv_pred(&recv_req_pred, &status);
            block_scatter_interm_stage(k++, first_elem, BLOCKSIZE, s, t, h, A,
                                       b, b_hat, c, y, err, dy, w);
            start_recv_pred(w[k], first_elem, BLOCKSIZE, k, &recv_req_pred);
            complete_send_pred(&send_req_pred, &status);
            start_send_pred(w[k], first_elem, BLOCKSIZE, k, &send_req_pred);
          }

          k = s - num_blocks;

          complete_recv_succ(&recv_req_succ, &status);
          block_scatter_interm_stage(k++, last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                     s, t, h, A, b, b_hat, c, y, err, dy, w);
          start_recv_succ(w[k], last_elem, BLOCKSIZE, k, &recv_req_succ);
          complete_send_succ(&send_req_succ, &status);
          start_send_succ(w[k], last_elem, BLOCKSIZE, k, &send_req_succ);

          for (i = last_elem - 2 * BLOCKSIZE + 1; i > first_elem;
               i -= BLOCKSIZE)
            block_scatter_interm_stage(k++, i, BLOCKSIZE, s, t, h, A, b, b_hat,
                                       c, y, err, dy, w);
        }

        complete_recv_pred(&recv_req_pred, &status);
        block_scatter_last_stage(first_elem, BLOCKSIZE, s, t, h, b, b_hat, c, y,
                                 err, dy, w, &my_err_max);
        complete_send_pred(&send_req_pred, &status);

        /* send first block of y to previous processor and
           start receive operation for the last block of the
           previous processor, but save a copy beforehand */

        copy_vector(y_old + first_elem_y, y + first_elem_y,
                    first_elem - first_elem_y);

        start_recv_pred(y, first_elem, BLOCKSIZE, 0, &recv_req_pred);
        start_send_pred(y, first_elem, BLOCKSIZE, 0, &send_req_pred);

        /* triangle (1, s-1) -- (num_blocks-1, s-num_blocks+1) -- (s-1, s-1) */

        for (j = s - num_blocks + 1; j < s - 1; j++)
        {
          k = j + 1;

          complete_recv_succ(&recv_req_succ, &status);
          block_scatter_interm_stage(j, last_elem - BLOCKSIZE + 1, BLOCKSIZE, s,
                                     t, h, A, b, b_hat, c, y, err, dy, w);
          start_recv_succ(w[k], last_elem, BLOCKSIZE, k, &recv_req_succ);
          complete_send_succ(&send_req_succ, &status);
          start_send_succ(w[k], last_elem, BLOCKSIZE, k, &send_req_succ);


          for (i = last_elem - 2 * BLOCKSIZE + 1; k < s - 1; i -= BLOCKSIZE)
            block_scatter_interm_stage(k++, i, BLOCKSIZE, s, t, h, A, b, b_hat,
                                       c, y, err, dy, w);

          block_scatter_last_stage(i, BLOCKSIZE, s, t, h, b, b_hat, c, y, err,
                                   dy, w, &my_err_max);
        }

        complete_recv_succ(&recv_req_succ, &status);
        block_scatter_last_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t, h,
                                 b, b_hat, c, y, err, dy, w, &my_err_max);
        complete_send_succ(&send_req_succ, &status);
      }
      else                      /* odd processor */
      {
        /* send first block of y to previous processor and
           start receive operation for the first block of the
           previous processor */

        start_recv_pred(y, first_elem, BLOCKSIZE, 0, &recv_req_pred);
        start_send_pred(y, first_elem, BLOCKSIZE, 0, &send_req_pred);

        /* triangle (0, 1) -- (num_blocks-1, 0) --
           (num_blocks-1, num_blocks-2) */

        block_scatter_first_stage_reverse(last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                          s, t, h, A, b, b_hat, c, y, err, dy,
                                          w);
        start_recv_succ(w[1], last_elem, BLOCKSIZE, 1, &recv_req_succ);
        start_send_succ(w[1], last_elem, BLOCKSIZE, 1, &send_req_succ);


        for (j = last_elem - 2 * BLOCKSIZE + 1; j > first_elem; j -= BLOCKSIZE)
        {
          block_scatter_first_stage_reverse(j, BLOCKSIZE, s, t, h, A, b, b_hat,
                                            c, y, err, dy, w);

          for (i = j + BLOCKSIZE, k = 1; i < last_elem - BLOCKSIZE + 1;
               i += BLOCKSIZE)
            block_scatter_interm_stage_reverse(k++, i, BLOCKSIZE, s, t, h, A, b,
                                               b_hat, c, y, err, dy, w);

          complete_recv_succ(&recv_req_succ, &status);
          block_scatter_interm_stage_reverse(k++, last_elem - BLOCKSIZE + 1,
                                             BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                             y, err, dy, w);
          start_recv_succ(w[k], last_elem, BLOCKSIZE, k, &recv_req_succ);
          complete_send_succ(&send_req_succ, &status);
          start_send_succ(w[k], last_elem, BLOCKSIZE, k, &send_req_succ);

        }

        /* parallelogram (0, 0) -- (num_blocks-1, num_blocks-1) --
           (num_block-1, s-1) -- (0, s-num_blocks) */

        complete_recv_pred(&recv_req_pred, &status);

        block_scatter_first_stage_reverse(first_elem, BLOCKSIZE, s, t, h, A, b,
                                          b_hat, c, y, err, dy, w);
        start_recv_pred(w[1], first_elem, BLOCKSIZE, 1, &recv_req_pred);
        complete_send_pred(&send_req_pred, &status);
        start_send_pred(w[1], first_elem, BLOCKSIZE, 1, &send_req_pred);

        for (i = first_elem + BLOCKSIZE, k = 1; i < last_elem - BLOCKSIZE + 1;
             i += BLOCKSIZE)
          block_scatter_interm_stage_reverse(k++, i, BLOCKSIZE, s, t, h, A, b,
                                             b_hat, c, y, err, dy, w);

        if (num_blocks < s)
        {
          complete_recv_succ(&recv_req_succ, &status);

          block_scatter_interm_stage_reverse(num_blocks - 1,
                                             last_elem - BLOCKSIZE + 1,
                                             BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                             y, err, dy, w);
          start_recv_succ(w[num_blocks], last_elem, BLOCKSIZE, num_blocks,
                          &recv_req_succ);
          complete_send_succ(&send_req_succ, &status);
          start_send_succ(w[num_blocks], last_elem, BLOCKSIZE, num_blocks,
                          &send_req_succ);

          for (j = 1; j < s - num_blocks; j++)
          {
            k = j + 1;

            complete_recv_pred(&recv_req_pred, &status);
            block_scatter_interm_stage_reverse(j, first_elem, BLOCKSIZE, s, t,
                                               h, A, b, b_hat, c, y, err, dy,
                                               w);
            start_recv_pred(w[k], first_elem, BLOCKSIZE, k, &recv_req_pred);
            complete_send_pred(&send_req_pred, &status);
            start_send_pred(w[k], first_elem, BLOCKSIZE, k, &send_req_pred);

            for (i = first_elem + BLOCKSIZE; i < last_elem - BLOCKSIZE + 1;
                 i += BLOCKSIZE)
              block_scatter_interm_stage_reverse(k++, i, BLOCKSIZE, s, t, h, A,
                                                 b, b_hat, c, y, err, dy, w);

            complete_recv_succ(&recv_req_succ, &status);
            block_scatter_interm_stage(k++, last_elem - BLOCKSIZE + 1,
                                       BLOCKSIZE, s, t, h, A, b, b_hat, c, y,
                                       err, dy, w);
            start_recv_succ(w[k], last_elem, BLOCKSIZE, k, &recv_req_succ);
            complete_send_succ(&send_req_succ, &status);
            start_send_succ(w[k], last_elem, BLOCKSIZE, k, &send_req_succ);
          }

          k = s - num_blocks;

          complete_recv_pred(&recv_req_pred, &status);
          block_scatter_interm_stage_reverse(k++, first_elem, BLOCKSIZE, s, t,
                                             h, A, b, b_hat, c, y, err, dy, w);
          start_recv_pred(w[k], first_elem, BLOCKSIZE, k, &recv_req_pred);
          complete_send_pred(&send_req_pred, &status);
          start_send_pred(w[k], first_elem, BLOCKSIZE, k, &send_req_pred);

          for (i = first_elem + BLOCKSIZE; i < last_elem - BLOCKSIZE + 1;
               i += BLOCKSIZE)
            block_scatter_interm_stage_reverse(k++, i, BLOCKSIZE, s, t, h, A, b,
                                               b_hat, c, y, err, dy, w);
        }

        complete_recv_succ(&recv_req_succ, &status);
        block_scatter_last_stage_reverse(last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                         s, t, h, b, b_hat, c, y, err, dy, w,
                                         &my_err_max);
        complete_send_succ(&send_req_succ, &status);

        /* send last block of y to next processor and
           start receive operation for the first block of the
           next processor, but save a copy beforehand */

        copy_vector(y_old + last_elem + 1, y + last_elem + 1,
                    last_elem_y - last_elem);

        start_recv_succ(y, last_elem, BLOCKSIZE, 0, &recv_req_succ);
        start_send_succ(y, last_elem, BLOCKSIZE, 0, &send_req_succ);

        /* triangle (num_blocks-2, s-1) -- (0, s-num_blocks+1) -- (0, s-1) */

        for (j = s - num_blocks + 1; j < s - 1; j++)
        {
          k = j + 1;

          complete_recv_pred(&recv_req_pred, &status);
          block_scatter_interm_stage_reverse(j, first_elem, BLOCKSIZE, s, t, h,
                                             A, b, b_hat, c, y, err, dy, w);
          start_recv_pred(w[k], first_elem, BLOCKSIZE, k, &recv_req_pred);
          complete_send_pred(&send_req_pred, &status);
          start_send_pred(w[k], first_elem, BLOCKSIZE, k, &send_req_pred);

          for (i = first_elem + BLOCKSIZE; k < s - 1; i += BLOCKSIZE)
            block_scatter_interm_stage_reverse(k++, i, BLOCKSIZE, s, t, h, A, b,
                                               b_hat, c, y, err, dy, w);

          block_scatter_last_stage_reverse(i, BLOCKSIZE, s, t, h, b, b_hat, c,
                                           y, err, dy, w, &my_err_max);
        }

        complete_recv_pred(&recv_req_pred, &status);
        block_scatter_last_stage_reverse(first_elem, BLOCKSIZE, s, t, h, b,
                                         b_hat, c, y, err, dy, w, &my_err_max);
        complete_send_pred(&send_req_pred, &status);
      }
    }
    else
    {
      if (me_is_even)
      {
        /* send last block of y to next processor and
           start receive operation for the first block of the
           next processor */

        start_recv_succ(y, last_elem, BLOCKSIZE, 0, &recv_req_succ);
        start_send_succ(y, last_elem, BLOCKSIZE, 0, &send_req_succ);

        /* initialize the pipeline */

        block_scatter_first_stage(first_elem, BLOCKSIZE, s, t, h, A, b, b_hat,
                                  c, y, err, dy, w);

        start_recv_pred(w[1], first_elem, BLOCKSIZE, 1, &recv_req_pred);
        start_send_pred(w[1], first_elem, BLOCKSIZE, 1, &send_req_pred);

        for (j = 1; j < s - 1; j++)
        {
          block_scatter_first_stage(first_elem + j * BLOCKSIZE, BLOCKSIZE, s, t,
                                    h, A, b, b_hat, c, y, err, dy, w);
          for (i = 1; i < j; i++)
            block_scatter_interm_stage(i, first_elem + (j - i) * BLOCKSIZE,
                                       BLOCKSIZE, s, t, h, A, b, b_hat, c, y,
                                       err, dy, w);

          complete_recv_pred(&recv_req_pred, &status);

          block_scatter_interm_stage(j, first_elem, BLOCKSIZE, s, t, h, A, b,
                                     b_hat, c, y, err, dy, w);

          start_recv_pred(w[j + 1], first_elem, BLOCKSIZE, j + 1,
                          &recv_req_pred);
          complete_send_pred(&send_req_pred, &status);
          start_send_pred(w[j + 1], first_elem, BLOCKSIZE, j + 1,
                          &send_req_pred);
        }

        block_scatter_first_stage(first_elem + (s - 1) * BLOCKSIZE, BLOCKSIZE,
                                  s, t, h, A, b, b_hat, c, y, err, dy, w);
        for (i = 1; i < j; i++)
          block_scatter_interm_stage(i, first_elem + (s - 1 - i) * BLOCKSIZE,
                                     BLOCKSIZE, s, t, h, A, b, b_hat, c, y, err,
                                     dy, w);

        complete_recv_pred(&recv_req_pred, &status);

        block_scatter_last_stage(first_elem, BLOCKSIZE, s, t, h, b, b_hat, c, y,
                                 err, dy, w, &my_err_max);

        complete_send_pred(&send_req_pred, &status);

        /* send first block of y to previous processor and
           start receive operation for the last block of the
           previous processor, but save a copy beforehand */

        copy_vector(y_old + first_elem_y, y + first_elem_y,
                    first_elem - first_elem_y);

        start_recv_pred(y, first_elem, BLOCKSIZE, 0, &recv_req_pred);
        start_send_pred(y, first_elem, BLOCKSIZE, 0, &send_req_pred);

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
                                   b, b_hat, c, y, err, dy, w, &my_err_max);
        }

        /* finalize the pipeline on the side with higher index */

        complete_recv_succ(&recv_req_succ, &status);

        block_scatter_first_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t, h,
                                  A, b, b_hat, c, y, err, dy, w);

        start_recv_succ(w[1], last_elem, BLOCKSIZE, 1, &recv_req_succ);
        complete_send_succ(&send_req_succ, &status);
        start_send_succ(w[1], last_elem, BLOCKSIZE, 1, &send_req_succ);


        for (i = 1; i < s - 1; i++)
          block_scatter_interm_stage(i,
                                     last_elem - BLOCKSIZE + 1 - i * BLOCKSIZE,
                                     BLOCKSIZE, s, t, h, A, b, b_hat, c, y, err,
                                     dy, w);

        block_scatter_last_stage(last_elem - BLOCKSIZE + 1 -
                                 (s - 1) * BLOCKSIZE, BLOCKSIZE, s, t, h, b,
                                 b_hat, c, y, err, dy, w, &my_err_max);

        for (j = 1; j < s - 1; j++)
        {
          complete_recv_succ(&recv_req_succ, &status);

          block_scatter_interm_stage(j, last_elem - BLOCKSIZE + 1, BLOCKSIZE, s,
                                     t, h, A, b, b_hat, c, y, err, dy, w);

          start_recv_succ(w[j + 1], last_elem, BLOCKSIZE, j + 1,
                          &recv_req_succ);
          complete_send_succ(&send_req_succ, &status);
          start_send_succ(w[j + 1], last_elem, BLOCKSIZE, j + 1,
                          &send_req_succ);

          for (i = j + 1; i < s - 1; i++)
            block_scatter_interm_stage(i,
                                       last_elem - BLOCKSIZE + 1 - (i -
                                                                    j) *
                                       BLOCKSIZE, BLOCKSIZE, s, t, h, A, b,
                                       b_hat, c, y, err, dy, w);

          block_scatter_last_stage(last_elem - BLOCKSIZE + 1 -
                                   (s - 1 - j) * BLOCKSIZE, BLOCKSIZE, s, t, h,
                                   b, b_hat, c, y, err, dy, w, &my_err_max);
        }

        complete_recv_succ(&recv_req_succ, &status);

        block_scatter_last_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t, h,
                                 b, b_hat, c, y, err, dy, w, &my_err_max);

        complete_send_succ(&send_req_succ, &status);
      }
      else
      {
        /* send first block of y to previous processor and
           start receive operation for the first block of the
           previous processor */

        start_recv_pred(y, first_elem, BLOCKSIZE, 0, &recv_req_pred);
        start_send_pred(y, first_elem, BLOCKSIZE, 0, &send_req_pred);

        /* initialize the pipeline */

        block_scatter_first_stage_reverse(last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                          s, t, h, A, b, b_hat, c, y, err, dy,
                                          w);

        start_recv_succ(w[1], last_elem, BLOCKSIZE, 1, &recv_req_succ);
        start_send_succ(w[1], last_elem, BLOCKSIZE, 1, &send_req_succ);


        for (j = 1; j < s - 1; j++)
        {
          block_scatter_first_stage_reverse(last_elem - BLOCKSIZE + 1 -
                                            j * BLOCKSIZE, BLOCKSIZE, s, t, h,
                                            A, b, b_hat, c, y, err, dy, w);
          for (i = 1; i < j; i++)
            block_scatter_interm_stage_reverse(i,
                                               last_elem - BLOCKSIZE + 1 - (j -
                                                                            i) *
                                               BLOCKSIZE, BLOCKSIZE, s, t, h, A,
                                               b, b_hat, c, y, err, dy, w);

          complete_recv_succ(&recv_req_succ, &status);
          block_scatter_interm_stage_reverse(j, last_elem - BLOCKSIZE + 1,
                                             BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                             y, err, dy, w);

          start_recv_succ(w[j + 1], last_elem, BLOCKSIZE, j + 1,
                          &recv_req_succ);
          complete_send_succ(&send_req_succ, &status);
          start_send_succ(w[j + 1], last_elem, BLOCKSIZE, j + 1,
                          &send_req_succ);
        }

        block_scatter_first_stage_reverse(last_elem - BLOCKSIZE + 1 -
                                          (s - 1) * BLOCKSIZE, BLOCKSIZE, s, t,
                                          h, A, b, b_hat, c, y, err, dy, w);
        for (i = 1; i < j; i++)
          block_scatter_interm_stage_reverse(i,
                                             last_elem - BLOCKSIZE + 1 - (s -
                                                                          1 -
                                                                          i) *
                                             BLOCKSIZE, BLOCKSIZE, s, t, h, A,
                                             b, b_hat, c, y, err, dy, w);

        complete_recv_succ(&recv_req_succ, &status);

        block_scatter_last_stage_reverse(last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                         s, t, h, b, b_hat, c, y, err, dy, w,
                                         &my_err_max);

        complete_send_succ(&send_req_succ, &status);

        /* send last block of y to next processor and
           start receive operation for the first block of the
           next processor, but save a copy beforehand */

        copy_vector(y_old + last_elem + 1, y + last_elem + 1,
                    last_elem_y - last_elem);

        start_recv_succ(y, last_elem, BLOCKSIZE, 0, &recv_req_succ);
        start_send_succ(y, last_elem, BLOCKSIZE, 0, &send_req_succ);

        /* sweep */

        for (j = last_elem - BLOCKSIZE + 1 - s * BLOCKSIZE; j > first_elem;
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
                                           &my_err_max);
        }

        /* finalize the pipeline on the side with lower index */

        complete_recv_pred(&recv_req_pred, &status);

        block_scatter_first_stage_reverse(first_elem, BLOCKSIZE, s, t, h, A, b,
                                          b_hat, c, y, err, dy, w);

        start_recv_pred(w[1], first_elem, BLOCKSIZE, 1, &recv_req_pred);
        complete_send_pred(&send_req_pred, &status);
        start_send_pred(w[1], first_elem, BLOCKSIZE, 1, &send_req_pred);

        for (i = 1; i < s - 1; i++)
          block_scatter_interm_stage_reverse(i, first_elem + i * BLOCKSIZE,
                                             BLOCKSIZE, s, t, h, A, b, b_hat, c,
                                             y, err, dy, w);

        block_scatter_last_stage_reverse(first_elem + (s - 1) * BLOCKSIZE,
                                         BLOCKSIZE, s, t, h, b, b_hat, c, y,
                                         err, dy, w, &my_err_max);

        for (j = 1; j < s - 1; j++)
        {
          complete_recv_pred(&recv_req_pred, &status);

          block_scatter_interm_stage_reverse(j, first_elem, BLOCKSIZE, s, t, h,
                                             A, b, b_hat, c, y, err, dy, w);

          start_recv_pred(w[j + 1], first_elem, BLOCKSIZE, j + 1,
                          &recv_req_pred);
          complete_send_pred(&send_req_pred, &status);
          start_send_pred(w[j + 1], first_elem, BLOCKSIZE, j + 1,
                          &send_req_pred);

          for (i = j + 1; i < s - 1; i++)
            block_scatter_interm_stage_reverse(i,
                                               first_elem + (i - j) * BLOCKSIZE,
                                               BLOCKSIZE, s, t, h, A, b, b_hat,
                                               c, y, err, dy, w);

          block_scatter_last_stage_reverse(first_elem + (s - 1 - j) * BLOCKSIZE,
                                           BLOCKSIZE, s, t, h, b, b_hat, c, y,
                                           err, dy, w, &my_err_max);
        }

        complete_recv_pred(&recv_req_pred, &status);

        block_scatter_last_stage_reverse(first_elem, BLOCKSIZE, s, t, h, b,
                                         b_hat, c, y, err, dy, w, &my_err_max);

        complete_send_pred(&send_req_pred, &status);
      }
    }

    /* step control */

    MPI_Allreduce(&my_err_max, &err_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if (me_is_even)
    {
      complete_recv_pred(&recv_req_pred, &status);
      complete_send_pred(&send_req_pred, &status);
    }
    else
    {
      complete_recv_succ(&recv_req_succ, &status);
      complete_send_succ(&send_req_succ, &status);
    }

    step_control(&t, &h, err_max, ord, tol, y + first_elem_y,
                 y_old + first_elem_y, num_elems_y, &steps_acc, &steps_rej);
  }

  timer_stop(&timer);

  copy_vector(dy, y + first_elem, num_elems);

  MPI_Gatherv(dy, num_elems, MPI_DOUBLE,
              y, elem_length, elem_offset, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  free_emb_rk_method(&A, &b, &b_hat, &c, s);

  FREE2D(w);
  FREE(err);
  FREE(dy);

  FREE(elem_offset);
  FREE(elem_length);

  FREE(block_offset);
  FREE(block_length);

  print_statistics(timer, steps_acc, steps_rej);
}

/******************************************************************************/
