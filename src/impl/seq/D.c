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
  double **w, *y_old, *err_vec, *delta_y;  
  double **A, *b, *b_hat, *bbs, *c;
  double F, error_max;
  int s, ord;
  double h, t, H = te - t0;
  double timer;
  int steps_acc = 0, steps_rej = 0;

  printf("Solver type: sequential embedded Runge-Kutta method\n");
  printf("Implementation variant: D (temporal locality of reads)\n");

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  bbs = MALLOC(s, double);
  for (i = 0; i < s; ++i) bbs[i] = b[i] - b_hat[i];

  ALLOC2D(w, s, ode_size, double);

  err_vec  = MALLOC(ode_size, double);
  delta_y  = MALLOC(ode_size, double);

  y_old = delta_y;

  h = initial_stepsize(t0, H, y0, ord, tol);

  copy_vector(y, y0, ode_size);

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {    
    for (i = 0; i < ode_size; ++i) 
    {
      double Y = y[i];

      F = h * ode_eval_comp(i, t + c[0] * h, y);     

      for (l = 1; l < s; l++) w[l][i] = Y + A[l][0] * F;      

      delta_y[i] = b[0]   * F;
      err_vec[i]  = bbs[0] * F; 
    }

    for (j = 1; j < s; ++j) 
    { 
      for (i = 0; i < ode_size; ++i) 
      {
        F = h * ode_eval_comp(i, t + c[j] * h, w[j]);

        for (l = j+1; l < s; l++) w[l][i] += A[l][j] * F;

        delta_y[i] += b[j]   * F;
        err_vec[i]  += bbs[j] * F; 
      }
    }

    error_max = 0.0;
    for (j = 0; j < ode_size; j++)
    {
      double yj_old;

      yj_old = y[j];
      y[j] += h * delta_y[j];
      y_old[j] = yj_old;        /* y_old and delta_y occupy the same space */

      update_error_max(&error_max, err_vec[j], y[j], yj_old);
    }

    /* step control */

    step_control(&t, &h, error_max, ord, tol, y, y_old, ode_size, &steps_acc,
                 &steps_rej);
  }

  timer_stop(&timer);

  free_emb_rk_method(&A, &b, &b_hat, &c, s);

  FREE2D(w);
  FREE(err_vec);
  FREE(delta_y);

  FREE(bbs);

  print_statistics(timer, steps_acc, steps_rej);
}

