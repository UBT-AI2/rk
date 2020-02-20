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
  int i, j, l;

  double **v, *y_old, *err, *w, *dy;
  double **A, *b, *b_hat, *c;
  double err_max;
  int s, ord;
  double h, t;
  double timer;
  int steps_acc = 0, steps_rej = 0;

  printf("Solver type: sequential embedded Runge-Kutta method\n");
  printf("Implementation variant: A (spatial locality)\n");

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  for (i = 0; i < s; ++i)
    b_hat[i] = b[i] - b_hat[i];

  ALLOC2D(v, s, ode_size, double);

  dy = MALLOC(ode_size, double);
  err = MALLOC(ode_size, double);

  w = y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y, y0, ode_size);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    /* stages */

    for (j = 0; j < ode_size; j++)
      v[0][j] = h * ode_eval_comp(j, t + c[0] * h, y);

    for (l = 1; l < s; l++)
    {
      for (j = 0; j < ode_size; j++)
        w[j] = y[j] + A[l][0] * v[0][j];

      for (i = 1; i < l; i++)
        for (j = 0; j < ode_size; j++)
          w[j] += A[l][i] * v[i][j];

      for (j = 0; j < ode_size; j++)
        v[l][j] = h * ode_eval_comp(j, t + c[l] * h, w);
    }

    /* output approximation */

    for (j = 0; j < ode_size; j++)
    {
      err[j] = b_hat[0] * v[0][j];
      dy[j] = b[0] * v[0][j];
    }

    for (i = 1; i < s; i++)
      for (j = 0; j < ode_size; j++)
      {
        err[j] += b_hat[i] * v[i][j];
        dy[j] += b[i] * v[i][j];
      }

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

  FREE2D(v);
  FREE(dy);
  FREE(err);

  print_statistics(timer, steps_acc, steps_rej);
}

/******************************************************************************/
