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

void solver(double t0, double te, double *y0, double *y, double tol)
{
  int i;
  double **w, *y_old, *err, *dy, *v;
  double **A, *b, *b_hat, *c;
  int **iz_A, *iz_b, *iz_b_hat, *iz_c;
  double **hA, *hb, *hb_hat, *hc;
  double err_max, my_err_max;
  int s, ord;
  double h, t;
  double timer;
  int steps_acc = 0, steps_rej = 0;
  int first_elem, num_elems, last_elem, *elem_offset, *elem_length;
  int num_blocks, *block_offset, *block_length;
  MPI_Request recv_req_succ, send_req_succ, recv_req_pred, send_req_pred;
  MPI_Status status;

  if (me == 0)
  {
    printf("Solver type: parallel embedded Runge-Kutta method");
    printf(" for distributed address space\n");
    printf
      ("Implementation variant: Dbcblock (Dblock with block-based communication)\n");
    printf("Number of MPI processes: %d\n", processes);
  }

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  for (i = 0; i < s; i++)
    b_hat[i] = b[i] - b_hat[i];

  alloc_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);
  zero_pattern(A, b, b_hat, c, iz_A, iz_b, iz_b_hat, iz_c, s);
  alloc_emb_rk_method(&hA, &hb, &hb_hat, &hc, s);

  ALLOC2D(w, s, ode_size, double);

  v = MALLOC(BLOCKSIZE, double);

  err = MALLOC(ode_size, double);
  dy = MALLOC(ode_size, double);

  elem_offset = MALLOC(processes, int);
  elem_length = MALLOC(processes, int);

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

  assert(s >= 2);               /* !!! at least two stages !!! */
  assert(num_blocks >= 2);      /* !!! at least 2 blocks per process !!! */

  y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y + first_elem, y0 + first_elem, num_elems);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    premult(h, A, b, b_hat, c, hA, hb, hb_hat, hc, s);

    my_err_max = 0.0;

    /* send last block of y to next processor and 
       start receive operation for the first block of the 
       next processor */

    start_recv_succ(y, last_elem, BLOCKSIZE, 0, &recv_req_succ);
    start_send_succ(y, last_elem, BLOCKSIZE, 0, &send_req_succ);

    /* send first block of y to previous processor and 
       start receive operation for the first block of the 
       previous processor */

    start_recv_pred(y, first_elem, BLOCKSIZE, 0, &recv_req_pred);
    start_send_pred(y, first_elem, BLOCKSIZE, 0, &send_req_pred);

    /* evaluate the inner blocks of the first stage */

    tiled_block_scatter_first_stage(first_elem + BLOCKSIZE,
                                    num_elems - 2 * BLOCKSIZE, s, t, h, hA,
                                    iz_A, hb, hb_hat, hc, y, err, dy, w, v);

    /* evaluate first block of the first stage and send 
       result to the previous processor */

    complete_recv_pred(&recv_req_pred, &status);
    start_recv_pred(w[1], first_elem, BLOCKSIZE, 1, &recv_req_pred);

    tiled_block_scatter_first_stage(first_elem, BLOCKSIZE, s, t, h, hA, iz_A,
                                    hb, hb_hat, hc, y, err, dy, w, v);

    complete_send_pred(&send_req_pred, &status);
    start_send_pred(w[1], first_elem, BLOCKSIZE, 1, &send_req_pred);

    /* evaluate last block of the second stage and send 
       result to the next processor */

    complete_recv_succ(&recv_req_succ, &status);
    start_recv_succ(w[1], last_elem, BLOCKSIZE, 1, &recv_req_succ);

    tiled_block_scatter_first_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t,
                                    h, hA, iz_A, hb, hb_hat, hc, y, err, dy, w,
                                    v);

    complete_send_succ(&send_req_succ, &status);
    start_send_succ(w[1], last_elem, BLOCKSIZE, 1, &send_req_succ);

    for (i = 1; i < s - 1; i++)
    {
      /* evaluate the inner blocks of stage i */

      tiled_block_scatter_interm_stage(i, first_elem + BLOCKSIZE,
                                       num_elems - 2 * BLOCKSIZE, s, t, h, hA,
                                       hb, hb_hat, hc, iz_A, iz_b, iz_b_hat, y,
                                       err, dy, w, v);

      /* evaluate first block of stage i and send 
         result to the previous processor */

      complete_recv_pred(&recv_req_pred, &status);
      start_recv_pred(w[i + 1], first_elem, BLOCKSIZE, i + 1, &recv_req_pred);

      tiled_block_scatter_interm_stage(i, first_elem, BLOCKSIZE, s, t, h, hA,
                                       hb, hb_hat, hc, iz_A, iz_b, iz_b_hat, y,
                                       err, dy, w, v);

      complete_send_pred(&send_req_pred, &status);
      start_send_pred(w[i + 1], first_elem, BLOCKSIZE, i + 1, &send_req_pred);

      /* evaluate last block of stage i and send 
         result to the next processor */

      complete_recv_succ(&recv_req_succ, &status);
      start_recv_succ(w[i + 1], last_elem, BLOCKSIZE, i + 1, &recv_req_succ);

      tiled_block_scatter_interm_stage(i, last_elem - BLOCKSIZE + 1, BLOCKSIZE,
                                       s, t, h, hA, hb, hb_hat, hc, iz_A, iz_b,
                                       iz_b_hat, y, err, dy, w, v);

      complete_send_succ(&send_req_succ, &status);
      start_send_succ(w[i + 1], last_elem, BLOCKSIZE, i + 1, &send_req_succ);
    }

    /* evaluate the inner blocks of stage s-1 */

    tiled_block_scatter_last_stage(first_elem + BLOCKSIZE,
                                   num_elems - 2 * BLOCKSIZE, s, t, h, hb,
                                   hb_hat, hc, iz_b, iz_b_hat, y, err, dy, w, v,
                                   &my_err_max);

    /* evaluate first block of stage s-1 */

    complete_recv_pred(&recv_req_pred, &status);
    tiled_block_scatter_last_stage(first_elem, BLOCKSIZE, s, t, h, hb, hb_hat,
                                   hc, iz_b, iz_b_hat, y, err, dy, w, v,
                                   &my_err_max);

    complete_send_pred(&send_req_pred, &status);

    /* evaluate last block of stage s-1 */

    complete_recv_succ(&recv_req_succ, &status);
    tiled_block_scatter_last_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t,
                                   h, hb, hb_hat, hc, iz_b, iz_b_hat, y, err,
                                   dy, w, v, &my_err_max);

    complete_send_succ(&send_req_succ, &status);

    /* step control */

    MPI_Allreduce(&my_err_max, &err_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    step_control(&t, &h, err_max, ord, tol, y + first_elem, y_old + first_elem,
                 num_elems, &steps_acc, &steps_rej);
  }

  timer_stop(&timer);

  copy_vector(dy, y + first_elem, num_elems);

  MPI_Gatherv(dy, num_elems, MPI_DOUBLE,
              y, elem_length, elem_offset, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  free_emb_rk_method(&A, &b, &b_hat, &c, s);
  free_emb_rk_method(&hA, &hb, &hb_hat, &hc, s);
  free_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);

  FREE2D(w);
  FREE(err);
  FREE(dy);

  FREE(v);

  FREE(elem_offset);
  FREE(elem_length);

  FREE(block_offset);
  FREE(block_length);

  print_statistics(timer, steps_acc, steps_rej);
}

/******************************************************************************/
