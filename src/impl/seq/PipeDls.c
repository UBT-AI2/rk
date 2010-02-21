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
#include "pipe-inline.h"

/******************************************************************************/

void solver(double t0, double te, double *y0, double *y, double tol)
{ 
  int i, j, l, k1, k2, k4;
  double *buf, **w, *y_old, *err_vec, *delta_y;
  double **A, *b, *b_hat, *bbs, *c;
  double error_max;
  int s, ord;
  double h, t, H = te - t0;
  double timer;
  int steps_acc = 0, steps_rej = 0;

  printf("Solver type: sequential embedded Runge-Kutta method\n");
  printf("Implementation variant: PipeDls (low-storage pipelining scheme based on implementation D)\n");

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  bbs = MALLOC(s, double);
  for (i = 0; i < s; ++i) bbs[i] = b[i] - b_hat[i];

  buf = MALLOC(ode_size + (s*s + 5*s - 4) * BLOCKSIZE / 2, double);
  w = MALLOC(s, double *);

  delta_y = buf;
  err_vec = delta_y + s * BLOCKSIZE;
  w[1] = err_vec + 3 * BLOCKSIZE;

  for (i = 2; i < s; ++i)
    w[i] = w[i - 1] + (i + 2) * BLOCKSIZE;

  y_old = delta_y;

  h = initial_stepsize(t0, H, y0, ord, tol);

  copy_vector(y, y0, ode_size);

  k1 = (s - 1) * BLOCKSIZE;
  k2 = ode_size - BLOCKSIZE;
  k4 = k1 + BLOCKSIZE;

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {    
    /* initialization */

    error_max = 0.0;

    for (j = 0; j < k1; j += BLOCKSIZE) 
    {
      BKSTAGE0(j);

      l = j, i = 1;
      while (l > 0) 
      {
	l -= BLOCKSIZE;
	BKSTAGEn(l, i);
	i++;
      }
    }

    /* sweep */

    for (i = k1; i < ode_size; i += k4)
    {
      BKSTAGE0(i);
      i -= BLOCKSIZE;

      for (j = 1; j < s - 1; j++)
      { 
 	BKSTAGEn(i, j);
	i -= BLOCKSIZE;
      }

      BKSTAGEsm1(i);
    }

    /* finalization */
    
    for (i = 1; i < s; i++)
    {
      for (j = i, l = k2; j < s - 1; j++, l -= BLOCKSIZE)
	BKSTAGEn(l, j);
      BKSTAGEsm1(l);
    }

    /* step control */

    step_control(&t, &h, error_max, ord, tol, y, y_old, ode_size, &steps_acc,
                 &steps_rej);
  }

  timer_stop(&timer);

  free_emb_rk_method(&A, &b, &b_hat, &c, s);

  FREE(w);
  FREE(buf);
  FREE(bbs);

  print_statistics(timer, steps_acc, steps_rej);
}
