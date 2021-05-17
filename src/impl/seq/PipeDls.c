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
  int i, j, l, k1, k2, k4;
  double *buf, **w, *y_old, *err, *dy, *v;
  double **A, *b, *b_hat, *c;
  int **iz_A, *iz_b, *iz_b_hat, *iz_c;
  double **hA, *hb, *hb_hat, *hc;
  double err_max;
  int s, ord;
  double h, t;
  double timer;
  int steps_acc = 0, steps_rej = 0;

  printf("Solver type: sequential embedded Runge-Kutta method\n");
  printf("Implementation variant: PipeDls ");
  printf("(low-storage pipelining scheme based on implementation D)\n");

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  for (i = 0; i < s; ++i)
    b_hat[i] = b[i] - b_hat[i];

  alloc_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);
  zero_pattern(A, b, b_hat, c, iz_A, iz_b, iz_b_hat, iz_c, s);
  alloc_emb_rk_method(&hA, &hb, &hb_hat, &hc, s);

  buf = MALLOC(ode_size + (s * s + 5 * s - 4) * BLOCKSIZE / 2, double);
  w = MALLOC(s, double *);

  v = MALLOC(BLOCKSIZE, double);

  dy = buf;
  err = dy + s * BLOCKSIZE;
  w[1] = err + 3 * BLOCKSIZE;

  for (i = 2; i < s; ++i)
    w[i] = w[i - 1] + (i + 2) * BLOCKSIZE;

  y_old = dy;

  h = initial_stepsize(t0, te - t0, y0, ord, tol);

  copy_vector(y, y0, ode_size);

  k1 = (s - 1) * BLOCKSIZE;
  k2 = ode_size - BLOCKSIZE;
  k4 = k1 + BLOCKSIZE;

  timer_start(&timer);

  FOR_ALL_GRIDPOINTS(t0, te, h, steps_acc, steps_rej)
  {
    premult(h, A, b, b_hat, c, hA, hb, hb_hat, hc, s);

    /* initialization */

    err_max = 0.0;

    for (j = 0; j < k1; j += BLOCKSIZE)
    {
      tiled_block_scatter_first_stage(j, BLOCKSIZE, s, t, h, hA, iz_A, hb,
                                      hb_hat, hc, y, err, dy, w, v);

      l = j, i = 1;
      while (l > 0)
      {
        l -= BLOCKSIZE;
        tiled_block_scatter_interm_stage(i, l, BLOCKSIZE, s, t, h, hA, hb,
                                         hb_hat, hc, iz_A, iz_b, iz_b_hat, y,
                                         err, dy, w, v);
        i++;
      }
    }

    /* sweep */

    for (i = k1; i < ode_size; i += k4)
    {
      tiled_block_scatter_first_stage(i, BLOCKSIZE, s, t, h, hA, iz_A, hb,
                                      hb_hat, hc, y, err, dy, w, v);
      i -= BLOCKSIZE;

      for (j = 1; j < s - 1; j++)
      {
        tiled_block_scatter_interm_stage(j, i, BLOCKSIZE, s, t, h, hA, hb,
                                         hb_hat, hc, iz_A, iz_b, iz_b_hat, y,
                                         err, dy, w, v);
        i -= BLOCKSIZE;
      }

      tiled_block_scatter_last_stage(i, BLOCKSIZE, s, t, h, hb, hb_hat, hc,
                                     iz_b, iz_b_hat, y, err, dy, w, v,
                                     &err_max);
    }

    /* finalization */

    for (i = 1; i < s; i++)
    {
      for (j = i, l = k2; j < s - 1; j++, l -= BLOCKSIZE)
        tiled_block_scatter_interm_stage(j, l, BLOCKSIZE, s, t, h, hA, hb,
                                         hb_hat, hc, iz_A, iz_b, iz_b_hat, y,
                                         err, dy, w, v);
      tiled_block_scatter_last_stage(l, BLOCKSIZE, s, t, h, hb, hb_hat, hc,
                                     iz_b, iz_b_hat, y, err, dy, w, v,
                                     &err_max);
    }

    /* step control */

    step_control(&t, &h, err_max, ord, tol, y, y_old, ode_size, &steps_acc,
                 &steps_rej);
  }

  timer_stop(&timer);

  free_emb_rk_method(&A, &b, &b_hat, &c, s);
  free_emb_rk_method(&hA, &hb, &hb_hat, &hc, s);
  free_zero_pattern(&iz_A, &iz_b, &iz_b_hat, &iz_c, s);

  FREE(w);
  FREE(buf);
  FREE(v);

  print_statistics(timer, steps_acc, steps_rej);
}
