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
  int i, j, l;

  double **v, *y_old, *err, *w_cur, *w_next, *dy;
  double **A, *b, *b_hat, *c;
  int **iz_A, *iz_b, *iz_b_hat, *iz_c;
  double **hA, *hb, *hb_hat, *hc;
  double err_max;
  int s, ord;
  double h, t;
  double timer;
  int steps_acc = 0, steps_rej = 0;

  printf("Solver type: sequential embedded Runge-Kutta method\n");
  printf("Implementation variant: Fblock");
  printf(" (temporal and spatial locality of writes with fused RHS)\n");

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  for (i = 0; i < s; ++i)
    b_hat[i] = b[i] - b_hat[i];

  ALLOC2D(v, s, ode_size, double);

  dy = MALLOC(ode_size, double);
  err = MALLOC(ode_size, double);

  w_cur = err;
  w_next = y_old = dy;

  alloc_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);
  zero_pattern(A, b, b_hat, c, iz_A, iz_b, iz_b_hat, iz_c, s);
  alloc_emb_rk_method(&hA, &hb, &hb_hat, &hc, s);

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y, y0, ode_size);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    premult(h, A, b, b_hat, c, hA, hb, hb_hat, hc, s);

    /* stages */

    tiled_block_rhs_gather_interm_stage(0, 0, ode_size, t, h, hA, iz_A, hc, y,
                                        y, w_next, v);

    for (l = 1; l < s - 1; l++)
    {
      swap_vectors(&w_cur, &w_next);
      tiled_block_rhs_gather_interm_stage(l, 0, ode_size, t, h, hA, iz_A, hc, y,
                                          w_cur, w_next, v);
    }

    block_rhs(l, 0, ode_size, t, h, hc, w_next, v);

    /* output approximation */

    tiled_block_gather_output(0, ode_size, s, hb, hb_hat, iz_b, iz_b_hat, err,
                              dy, v);

    err_max = 0.0;
    for (j = 0; j < ode_size; j++)
    {
      double yj_old = y[j];
      y[j] += dy[j];
      y_old[j] = yj_old;        /* y_old and dy occupy the same space */
      update_error_max(&err_max, err[j], y[j], yj_old);
    }

    /* step control */

    step_control(&t, &h, err_max, ord, tol, y, y_old, ode_size, &steps_acc,
                 &steps_rej);
  }

  timer_stop(&timer);

  free_emb_rk_method(&A, &b, &b_hat, &c, s);
  free_emb_rk_method(&hA, &hb, &hb_hat, &hc, s);
  free_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);

  FREE2D(v);
  FREE(dy);
  FREE(err);

  print_statistics(timer, steps_acc, steps_rej);
}

/******************************************************************************/
