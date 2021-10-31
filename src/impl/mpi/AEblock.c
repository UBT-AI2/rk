/******************************************************************************/
/* This file is part of a collection of embedded Runge-Kutta solvers.         */
/* Copyright (C) 2009-2021, Matthias Korch, University of Bayreuth, Germany.  */
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
  int i, j, l;
  double **v, *w, *y_old, *err, *dy, *gathered_w;
  double **A, *b, *b_hat, *c;
  int **iz_A, *iz_b, *iz_b_hat, *iz_c;
  double **hA, *hb, *hb_hat, *hc;
  double err_max, my_err_max;
  int s, ord;
  double h, t;
  double timer;
  int steps_acc = 0, steps_rej = 0;
  int first_elem, last_elem, num_elems, *elem_offset, *elem_length;

  if (me == 0)
  {
    printf("Solver type: parallel embedded Runge-Kutta method");
    printf(" for distributed address space\n");
    printf
      ("Implementation variant: AEblock (temporal and spatial locality of writes)\n");
    printf("Number of MPI processes: %d\n", processes);
  }

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  for (i = 0; i < s; i++)
    b_hat[i] = b[i] - b_hat[i];

  alloc_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);
  zero_pattern(A, b, b_hat, c, iz_A, iz_b, iz_b_hat, iz_c, s);
  alloc_emb_rk_method(&hA, &hb, &hb_hat, &hc, s);

  ALLOC2D(v, s, ode_size, double);

  dy = MALLOC(ode_size, double);
  err = MALLOC(ode_size, double);

  w = y_old = dy;
  gathered_w = err;

  elem_offset = MALLOC(processes, int);
  elem_length = MALLOC(processes, int);

  blockwise_distribution(processes, ode_size, elem_offset, elem_length);

  first_elem = elem_offset[me];
  num_elems = elem_length[me];
  last_elem = first_elem + num_elems - 1;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y + first_elem, y0 + first_elem, num_elems);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    premult(h, A, b, b_hat, c, hA, hb, hb_hat, hc, s);

    /* stages */

    MPI_Allgatherv(y + first_elem, num_elems, MPI_DOUBLE,
                   gathered_w, elem_length, elem_offset, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    block_rhs(0, first_elem, num_elems, t, h, hc, gathered_w, v);

    for (l = 1; l < s; l++)
    {
      tiled_block_gather_interm_stage(l, first_elem, num_elems, hA, iz_A, y, w,
                                      v);

      MPI_Allgatherv(w + first_elem, num_elems, MPI_DOUBLE,
                     gathered_w, elem_length, elem_offset, MPI_DOUBLE,
                     MPI_COMM_WORLD);

      block_rhs(l, first_elem, num_elems, t, h, hc, gathered_w, v);
    }

    /* output approximation */

    tiled_block_gather_output(first_elem, num_elems, s, hb, hb_hat, iz_b,
                              iz_b_hat, err, dy, v);

    my_err_max = 0.0;
    for (j = first_elem; j <= last_elem; j++)
    {
      double yj_old = y[j];
      y[j] += dy[j];
      y_old[j] = yj_old;        /* y_old and dy occupy the same space */
      update_error_max(&my_err_max, err[j], y[j], yj_old);
    }

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

  FREE2D(v);
  FREE(dy);
  FREE(err);

  FREE(elem_offset);
  FREE(elem_length);

  print_statistics(timer, steps_acc, steps_rej);
}

/******************************************************************************/
