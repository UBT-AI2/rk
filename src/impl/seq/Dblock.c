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
  double err_max;
  int s, ord;
  double h, t;
  double timer;
  int steps_acc = 0, steps_rej = 0;

  printf("Solver type: sequential embedded Runge-Kutta method\n");
  printf
    ("Implementation variant: Dblock (temporal and spatial locality of reads)\n");

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  for (i = 0; i < s; i++)
    b_hat[i] = b[i] - b_hat[i];

  alloc_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);
  zero_pattern(A, b, b_hat, c, iz_A, iz_b, iz_b_hat, iz_c, s);

  ALLOC2D(w, s, ode_size, double);

  v = MALLOC(BLOCKSIZE, double);

  err = MALLOC(ode_size, double);
  dy = MALLOC(ode_size, double);

  y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y, y0, ode_size);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    err_max = 0.0;

    tiled_block_scatter_first_stage(0, ode_size, s, t, h, A, iz_A, b, b_hat, c,
                                    y, err, dy, w, v);

    for (i = 1; i < s - 1; i++)
      tiled_block_scatter_interm_stage(i, 0, ode_size, s, t, h, A, b, b_hat, c,
                                       iz_A, iz_b, iz_b_hat, y, err, dy, w, v);

    tiled_block_scatter_last_stage(0, ode_size, s, t, h, b, b_hat, c, iz_b,
                                   iz_b_hat, y, err, dy, w, v, &err_max);

    /* step control */

    step_control(&t, &h, err_max, ord, tol, y, y_old, ode_size, &steps_acc,
                 &steps_rej);
  }

  timer_stop(&timer);

  free_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);

  FREE2D(w);
  FREE(err);
  FREE(dy);

  FREE(v);

  print_statistics(timer, steps_acc, steps_rej);
}

/******************************************************************************/
