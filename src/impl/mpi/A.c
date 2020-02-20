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
  double **v, *w, *y_old, *err, *dy, *gathered_w;
  double **A, *b, *b_hat, *c;
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
    printf("Implementation variant: A (spatial locality)\n");
    printf("Number of MPI processes: %d\n", processes);
  }

  METHOD(&A, &b, &b_hat, &c, &s, &ord);

  for (i = 0; i < s; i++)
    b_hat[i] = b[i] - b_hat[i];

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
    /* stages */

    MPI_Allgatherv(y + first_elem, num_elems, MPI_DOUBLE,
                   gathered_w, elem_length, elem_offset, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    for (j = first_elem; j <= last_elem; j++)
      v[0][j] = h * ode_eval_comp(j, t + c[0] * h, gathered_w);

    for (l = 1; l < s; l++)
    {
      for (j = first_elem; j <= last_elem; j++)
        w[j] = y[j] + A[l][0] * v[0][j];

      for (i = 1; i < l; i++)
        for (j = first_elem; j <= last_elem; j++)
          w[j] += A[l][i] * v[i][j];

      MPI_Allgatherv(w + first_elem, num_elems, MPI_DOUBLE,
                     gathered_w, elem_length, elem_offset, MPI_DOUBLE,
                     MPI_COMM_WORLD);

      for (j = first_elem; j <= last_elem; j++)
        v[l][j] = h * ode_eval_comp(j, t + c[l] * h, gathered_w);
    }

    /* output approximation */

    for (j = first_elem; j <= last_elem; j++)
    {
      err[j] = b_hat[0] * v[0][j];
      dy[j] = b[0] * v[0][j];
    }

    for (i = 1; i < s; i++)
      for (j = first_elem; j <= last_elem; j++)
      {
        err[j] += b_hat[i] * v[i][j];
        dy[j] += b[i] * v[i][j];
      }

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

  FREE2D(v);
  FREE(dy);
  FREE(err);

  FREE(elem_offset);
  FREE(elem_length);

  print_statistics(timer, steps_acc, steps_rej);
}

/******************************************************************************/
