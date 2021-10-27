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
  int i, j, k;
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
  MPI_Request send_req;
  MPI_Status status;
  double *sbuf, *rbuf;
  int bufsize, recv_ofs;

  if (me == 0)
  {
    printf("Solver type: parallel embedded Runge-Kutta method");
    printf(" for distributed address space\n");
    printf("Implementation variant: PipeD2 ");
    printf("(pipelining scheme based on implementation D ");
    printf("with alternative finalization strategy)\n");
    printf("Number of MPI processes: %d\n", processes);
  }

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  for (i = 0; i < s; i++)
    b_hat[i] = b[i] - b_hat[i];

  alloc_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);
  zero_pattern(A, b, b_hat, c, iz_A, iz_b, iz_b_hat, iz_c, s);
  alloc_emb_rk_method(&hA, &hb, &hb_hat, &hc, s);

  ALLOC2D(w, s, ode_size, double);

  err = MALLOC(ode_size, double);
  dy = MALLOC(ode_size, double);

  v = MALLOC(BLOCKSIZE, double);

  bufsize = (s * (s + 7) / 2 - 3) * BLOCKSIZE;
  sbuf = MALLOC(bufsize, double);
  rbuf = MALLOC(bufsize, double);

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

  recv_ofs = (me == processes - 1) ? 0 : last_elem + 1;

  assert(s >= 2);               /* !!! at least two stages !!! */
  assert(num_blocks >= 2 * s);  /* !!! at least 2s blocks per process !!! */

  y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y + first_elem, y0 + first_elem, num_elems);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    premult(h, A, b, b_hat, hc, hA, hb, hb_hat, hc, s);

    my_err_max = 0.0;

    /* initialize the pipeline */

    for (j = 1; j < s; j++)
    {
      tiled_block_scatter_first_stage(first_elem + (2 * j - 1) * BLOCKSIZE,
                                      BLOCKSIZE, s, t, h, hA, iz_A, hb, hb_hat,
                                      hc, y, err, dy, w, v);
      for (i = 1; i < j; i++)
        tiled_block_scatter_interm_stage(i,
                                         first_elem + (2 * j - 1 -
                                                       i) * BLOCKSIZE,
                                         BLOCKSIZE, s, t, h, hA, hb, hb_hat, hc,
                                         iz_A, iz_b, iz_b_hat, y, err, dy, w,
                                         v);

      tiled_block_scatter_first_stage(first_elem + 2 * j * BLOCKSIZE, BLOCKSIZE,
                                      s, t, h, hA, iz_A, hb, hb_hat, hc, y, err,
                                      dy, w, v);
      for (i = 1; i < j; i++)
        tiled_block_scatter_interm_stage(i,
                                         first_elem + (2 * j - i) * BLOCKSIZE,
                                         BLOCKSIZE, s, t, h, hA, hb, hb_hat, hc,
                                         iz_A, iz_b, iz_b_hat, y, err, dy, w,
                                         v);
    }

    /* send data */

    copy_vector(sbuf, y + first_elem, s * BLOCKSIZE);
    for (i = 1, k = s; i < s; k += i + 1, i++)
      copy_vector(sbuf + k * BLOCKSIZE, w[i] + first_elem + BLOCKSIZE,
                  (i + 1) * BLOCKSIZE);
    copy_vector(sbuf + k * BLOCKSIZE, err + first_elem + BLOCKSIZE,
                (s - 1) * BLOCKSIZE);
    k += (s - 1);
    copy_vector(sbuf + k * BLOCKSIZE, dy + first_elem + BLOCKSIZE,
                (s - 1) * BLOCKSIZE);

    MPI_Isend(sbuf, bufsize, MPI_DOUBLE, (me == 0) ? processes - 1 : me - 1, 0,
              MPI_COMM_WORLD, &send_req);

    /* sweep */

    for (j = first_elem + (2 * s - 1) * BLOCKSIZE;
         j < last_elem - BLOCKSIZE + 1; j += BLOCKSIZE)
    {
      tiled_block_scatter_first_stage(j, BLOCKSIZE, s, t, h, hA, iz_A, hb,
                                      hb_hat, hc, y, err, dy, w, v);
      for (i = 1; i < s - 1; i++)
        tiled_block_scatter_interm_stage(i, j - i * BLOCKSIZE, BLOCKSIZE, s, t,
                                         h, hA, hb, hb_hat, hc, iz_A, iz_b,
                                         iz_b_hat, y, err, dy, w, v);
      tiled_block_scatter_last_stage(j - ((s - 1) * BLOCKSIZE), BLOCKSIZE, s, t,
                                     h, hb, hb_hat, hc, iz_b, iz_b_hat, y, err,
                                     dy, w, v, &my_err_max);
    }

    /* receive data */

    MPI_Recv(rbuf, bufsize, MPI_DOUBLE, (me == processes - 1) ? 0 : me + 1, 0,
             MPI_COMM_WORLD, &status);

    copy_vector(y + recv_ofs, rbuf, s * BLOCKSIZE);
    for (i = 1, k = s; i < s; k += i + 1, i++)
      copy_vector(w[i] + recv_ofs + BLOCKSIZE, rbuf + k * BLOCKSIZE,
                  (i + 1) * BLOCKSIZE);
    copy_vector(err + recv_ofs + BLOCKSIZE, rbuf + k * BLOCKSIZE,
                (s - 1) * BLOCKSIZE);
    k += (s - 1);
    copy_vector(dy + recv_ofs + BLOCKSIZE, rbuf + k * BLOCKSIZE,
                (s - 1) * BLOCKSIZE);

    MPI_Wait(&send_req, &status);

    /* finalization */

    tiled_block_scatter_first_stage(last_elem - BLOCKSIZE + 1, BLOCKSIZE, s, t,
                                    h, hA, iz_A, hb, hb_hat, hc, y, err, dy, w,
                                    v);

    for (i = 1; i < s - 1; i++)
      tiled_block_scatter_interm_stage(i,
                                       last_elem - BLOCKSIZE + 1 -
                                       i * BLOCKSIZE, BLOCKSIZE, s, t, h, hA,
                                       hb, hb_hat, hc, iz_A, iz_b, iz_b_hat, y,
                                       err, dy, w, v);

    tiled_block_scatter_last_stage(last_elem - BLOCKSIZE + 1 -
                                   (s - 1) * BLOCKSIZE, BLOCKSIZE, s, t, h, hb,
                                   hb_hat, hc, iz_b, iz_b_hat, y, err, dy, w, v,
                                   &my_err_max);


    tiled_block_scatter_first_stage((last_elem + 1) % ode_size, BLOCKSIZE, s, t,
                                    h, hA, iz_A, hb, hb_hat, hc, y, err, dy, w,
                                    v);

    for (i = 1; i < s - 1; i++)
      tiled_block_scatter_interm_stage(i, last_elem + 1 - i * BLOCKSIZE,
                                       BLOCKSIZE, s, t, h, hA, hb, hb_hat, hc,
                                       iz_A, iz_b, iz_b_hat, y, err, dy, w, v);

    tiled_block_scatter_last_stage(last_elem + 1 - (s - 1) * BLOCKSIZE,
                                   BLOCKSIZE, s, t, h, hb, hb_hat, hc, iz_b,
                                   iz_b_hat, y, err, dy, w, v, &my_err_max);

    for (i = 1; i < s; i++)
    {
      for (j = i; j < s - 1; j++)
        tiled_block_scatter_interm_stage(j,
                                         (last_elem - BLOCKSIZE + 1 +
                                          (2 * i - j) * BLOCKSIZE) % ode_size,
                                         BLOCKSIZE, s, t, h, hA, hb, hb_hat, hc,
                                         iz_A, iz_b, iz_b_hat, y, err, dy, w,
                                         v);

      tiled_block_scatter_last_stage((last_elem - BLOCKSIZE + 1 +
                                      (2 * i - s + 1) * BLOCKSIZE) % ode_size,
                                     BLOCKSIZE, s, t, h, hb, hb_hat, hc, iz_b,
                                     iz_b_hat, y, err, dy, w, v, &my_err_max);

      for (j = i; j < s - 1; j++)
        tiled_block_scatter_interm_stage(j,
                                         (last_elem + 1 +
                                          (2 * i - j) * BLOCKSIZE) % ode_size,
                                         BLOCKSIZE, s, t, h, hA, hb, hb_hat, hc,
                                         iz_A, iz_b, iz_b_hat, y, err, dy, w,
                                         v);

      tiled_block_scatter_last_stage((last_elem + 1 +
                                      (2 * i - s + 1) * BLOCKSIZE) % ode_size,
                                     BLOCKSIZE, s, t, h, hb, hb_hat, hc, iz_b,
                                     iz_b_hat, y, err, dy, w, v, &my_err_max);
    }

    /* step control */

    MPI_Allreduce(&my_err_max, &err_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if (err_max <= tol || tol <= 0.0)
    {
      /* if the step was accepted, we have to send s blocks of y to
         the predecessor */

      MPI_Wait(&send_req, &status);

      if (me == processes - 1)
        MPI_Isend(y, s * BLOCKSIZE, MPI_DOUBLE,
                  0, 1, MPI_COMM_WORLD, &send_req);
      else
        MPI_Isend(y + last_elem + 1, s * BLOCKSIZE, MPI_DOUBLE,
                  me + 1, 1, MPI_COMM_WORLD, &send_req);

      MPI_Recv(y + first_elem, s * BLOCKSIZE, MPI_DOUBLE,
               me == 0 ? processes - 1 : me - 1, 1, MPI_COMM_WORLD, &status);

      MPI_Wait(&send_req, &status);
    }
    else
    {
      /* if the step was rejected, we have to send s blocks of y_old to
         the predecessor */

      if (me == processes - 1)
        MPI_Isend(y_old, s * BLOCKSIZE, MPI_DOUBLE,
                  0, 1, MPI_COMM_WORLD, &send_req);
      else
        MPI_Isend(y_old + last_elem + 1, s * BLOCKSIZE, MPI_DOUBLE,
                  me + 1, 1, MPI_COMM_WORLD, &send_req);

      MPI_Recv(y_old + first_elem, s * BLOCKSIZE, MPI_DOUBLE,
               me == 0 ? processes - 1 : me - 1, 1, MPI_COMM_WORLD, &status);

    }

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

  FREE(sbuf);
  FREE(rbuf);

  print_statistics(timer, steps_acc, steps_rej);
}

/******************************************************************************/
