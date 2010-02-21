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
  int i, j, l, k1, k2, k4;
  double **w, *y_old, *err, *dy;
  double **A, *b, *b_hat, *bbs, *c;
  double err_max;
  int s, ord;
  double h, t;
  double timer;
  int steps_acc = 0, steps_rej = 0;

  printf("Solver type: sequential embedded Runge-Kutta method\n");
  printf("Implementation variant: PipeD ");
  printf("(pipelining scheme based on implementation D)\n");

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  bbs = MALLOC(s, double);
  for (i = 0; i < s; ++i)
    bbs[i] = b[i] - b_hat[i];

  ALLOC2D(w, s, ode_size, double);

  err = MALLOC(ode_size, double);
  dy = MALLOC(ode_size, double);

  y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y, y0, ode_size);

  k1 = (s - 1) * BLOCKSIZE;
  k2 = ode_size - BLOCKSIZE;
  k4 = k1 + BLOCKSIZE;

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    /* initialization */

    err_max = 0.0;

    for (j = 0; j < k1; j += BLOCKSIZE)
    {
      block_first_stage(j, BLOCKSIZE, s, t, h, A, b, bbs, c, y, err, dy, w);

      l = j, i = 1;
      while (l > 0)
      {
        l -= BLOCKSIZE;
        block_interm_stage(i, l, BLOCKSIZE, s, t, h, A, b, bbs, c, y, err, dy,
                           w);
        i++;
      }
    }

    /* sweep */

    for (i = k1; i < ode_size; i += k4)
    {
      block_first_stage(i, BLOCKSIZE, s, t, h, A, b, bbs, c, y, err, dy, w);
      i -= BLOCKSIZE;

      for (j = 1; j < s - 1; j++)
      {
        block_interm_stage(j, i, BLOCKSIZE, s, t, h, A, b, bbs, c, y, err, dy,
                           w);
        i -= BLOCKSIZE;
      }

      block_last_stage(i, BLOCKSIZE, s, t, h, b, bbs, c, y, err, dy, w,
                       &err_max);
    }

    /* finalization */

    for (i = 1; i < s; i++)
    {
      for (j = i, l = k2; j < s - 1; j++, l -= BLOCKSIZE)
        block_interm_stage(j, l, BLOCKSIZE, s, t, h, A, b, bbs, c, y, err, dy,
                           w);
      block_last_stage(l, BLOCKSIZE, s, t, h, b, bbs, c, y, err, dy, w,
                       &err_max);
    }

    /* step control */

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
