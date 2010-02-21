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
  int i;
  double **w, *y_old, *err, *dy;
  double **A, *b, *b_hat, *bbs, *c;
  double err_max;
  int s, ord;
  double h, t;
  double timer;
  int steps_acc = 0, steps_rej = 0;

  printf("Solver type: sequential embedded Runge-Kutta method\n");
  printf("Implementation variant: D (temporal locality of reads)\n");

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  bbs = MALLOC(s, double);
  for (i = 0; i < s; i++)
    bbs[i] = b[i] - b_hat[i];

  ALLOC2D(w, s, ode_size, double);

  err = MALLOC(ode_size, double);
  dy = MALLOC(ode_size, double);

  y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y, y0, ode_size);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    printf("%f %f %e %e\n", t0, te, t, h);

    err_max = 0.0;

    block_first_stage(0, ode_size, s, t, h, A, b, bbs, c, y, err, dy, w);

    for (i = 1; i < s - 1; i++)
      block_interm_stage(i, 0, ode_size, s, t, h, A, b, bbs, c, y, err, dy, w);

    block_last_stage(0, ode_size, s, t, h, b, bbs, c, y, err, dy, w, &err_max);

    /* step control */

    printf("e: %e\n", err_max);

    step_control(&t, &h, err_max, ord, tol, y, y_old, ode_size, &steps_acc,
                 &steps_rej);
  }

  timer_stop(&timer);

  free_emb_rk_method(&A, &b, &b_hat, &c, s);

  FREE2D(w);
  FREE(err);
  FREE(dy);

  FREE(bbs);

  print_statistics(timer, steps_acc, steps_rej);
}
