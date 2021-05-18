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
  double **w, *y_old, *err, *dy, *gathered_w, *v;
  double **A, *b, *b_hat, *c;
  int **iz_A, *iz_b, *iz_b_hat, *iz_c;
  double err_max, my_err_max;
  int s, ord;
  double h, t;
  double timer;
  int steps_acc = 0, steps_rej = 0;
  int first_elem, num_elems, *elem_offset, *elem_length;

  if (me == 0)
  {
    printf("Solver type: parallel embedded Runge-Kutta method");
    printf(" for distributed address space\n");
    printf("Implementation variant: D (temporal locality of reads)\n");
    printf("Number of MPI processes: %d\n", processes);
  }

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  for (i = 0; i < s; i++)
    b_hat[i] = b[i] - b_hat[i];

  alloc_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);
  zero_pattern(A, b, b_hat, c, iz_A, iz_b, iz_b_hat, iz_c, s);

  ALLOC2D(w, s, ode_size, double);

  v = MALLOC(BLOCKSIZE, double);

  err = MALLOC(ode_size, double);
  dy = MALLOC(ode_size, double);
  gathered_w = w[0];

  elem_offset = MALLOC(processes, int);
  elem_length = MALLOC(processes, int);

  blockwise_distribution(processes, ode_size, elem_offset, elem_length);

  first_elem = elem_offset[me];
  num_elems = elem_length[me];

  assert(s >= 2);               /* !!! at least two stages !!! */

  y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y + first_elem, y0 + first_elem, num_elems);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    my_err_max = 0.0;

    /* first stage (0) */

    MPI_Allgatherv(y + first_elem, num_elems, MPI_DOUBLE,
                   gathered_w, elem_length, elem_offset, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    swap_vectors(&y, &gathered_w);
    tiled_block_scatter_first_stage(first_elem, num_elems, s, t, h, A, iz_A, b,
                                    b_hat, c, y, err, dy, w, v);
    swap_vectors(&y, &gathered_w);

    /* stage 1 to s-2 */

    for (i = 1; i < s - 1; i++)
    {
      MPI_Allgatherv(w[i] + first_elem, num_elems, MPI_DOUBLE,
                     gathered_w, elem_length, elem_offset, MPI_DOUBLE,
                     MPI_COMM_WORLD);

      swap_vectors(&w[i], &gathered_w);
      tiled_block_scatter_interm_stage(i, first_elem, num_elems, s, t, h, A, b,
                                       b_hat, c, iz_A, iz_b, iz_b_hat, y, err,
                                       dy, w, v);
      swap_vectors(&w[i], &gathered_w);

    }

    /* last stage (s-1) */

    MPI_Allgatherv(w[s - 1] + first_elem, num_elems, MPI_DOUBLE,
                   gathered_w, elem_length, elem_offset, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    swap_vectors(&w[s - 1], &gathered_w);
    tiled_block_scatter_last_stage(first_elem, num_elems, s, t, h, b, b_hat, c,
                                   iz_b, iz_b_hat, y, err, dy, w, v,
                                   &my_err_max);
    swap_vectors(&w[s - 1], &gathered_w);

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
  free_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);

  FREE2D(w);
  FREE(err);
  FREE(dy);

  FREE(v);

  FREE(elem_offset);
  FREE(elem_length);

  print_statistics(timer, steps_acc, steps_rej);
}

/******************************************************************************/
