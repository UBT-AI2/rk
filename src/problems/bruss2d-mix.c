/******************************************************************************/
/* This file is part of a collection of embedded Runge-Kutta solvers.         */
/* Copyright (C) 2009, Matthias Korch, University of Bayreuth, Germany.       */
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

/******************************************************************************/
/* Test equation: BRUSS2D-MIX                                                 */
/* Brusselator function with ordering {U11, V11, U12, V12, ...}               */
/******************************************************************************/

#include "ode.h"

/******************************************************************************/

int ode_size = 2 * BRUSS_GRID_SIZE * BRUSS_GRID_SIZE;

/******************************************************************************/
/* Initialization of the start vector (initial value)                         */
/******************************************************************************/

void ode_start(double t, double *y0)
{
  int i, j;

  printf("ODE: BRUSS2D-MIX ");
  printf("(2D Brusselator with mixed row-oriented ordering)\n");
  printf("Grid size: %d\n", BRUSS_GRID_SIZE);
  printf("Alpha: %.2e\n", BRUSS_ALPHA);

  for (i = 0; i < BRUSS_GRID_SIZE; i++)
    for (j = 0; j < BRUSS_GRID_SIZE; j++)
      y0[2 * BRUSS_GRID_SIZE * i + 2 * j] =
        0.5 + (double) j / (double) (BRUSS_GRID_SIZE - 1);

  for (i = 0; i < BRUSS_GRID_SIZE; i++)
    for (j = 0; j < BRUSS_GRID_SIZE; j++)
      y0[2 * BRUSS_GRID_SIZE * i + 2 * j + 1] =
        1.0 + (5.0 * (double) i) / (double) (BRUSS_GRID_SIZE - 1);
}

/******************************************************************************/
/* Evaluation of component i                                                  */
/******************************************************************************/

double ode_eval_comp(int i, double t, const double *y)
{
  double N1 = (double) BRUSS_GRID_SIZE - 1.0;
  int N2 = BRUSS_GRID_SIZE + BRUSS_GRID_SIZE;
  
  int k = i / N2;            /* row index */
  int j = (i - k * N2) / 2;  /* column index */
  int v = i & 1;             /* i even -> variable U; i odd -> variable V */ 

  if (!v)                    /* --- U --------------------------------------- */
  {
    if (k == 0)
    {
      if (j == 0)
        /* U(0,0) */
        return 1.0
          + y[i] * y[i] * y[i + 1] - 4.4 * y[i]
          + BRUSS_ALPHA * N1 * N1 * (2.0 * y[i + N2] + 2.0 * y[i + 2] -
                                     4.0 * y[i]);

      if (j == BRUSS_GRID_SIZE - 1)
        /* U(0,BRUSS_GRID_SIZE-1) */
        return 1.0
          + y[i] * y[i] * y[i + 1] - 4.4 * y[i]
          + BRUSS_ALPHA * N1 * N1 * (2.0 * y[i + N2] + 2.0 * y[i - 2] -
                                     4.0 * y[i]);

      /* U(0, j) */
      return 1.0
        + y[i] * y[i] * y[i + 1] - 4.4 * y[i]
        + BRUSS_ALPHA * N1 * N1
        * (2.0 * y[i + N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);
    }
    else if (k == BRUSS_GRID_SIZE - 1)
    {
      if (j == 0)
        /* U(BRUSS_GRID_SIZE-1,0) */
        return 1.0
          + y[i] * y[i] * y[i + 1] - 4.4 * y[i]
          + BRUSS_ALPHA * N1 * N1 * (2.0 * y[i - N2] + 2.0 * y[i + 2] -
                                     4.0 * y[i]);

      if (j == BRUSS_GRID_SIZE - 1)
        /* U(BRUSS_GRID_SIZE-1,BRUSS_GRID_SIZE-1) */
        return 1.0
          + y[i] * y[i] * y[i + 1] - 4.4 * y[i]
          + BRUSS_ALPHA * N1 * N1 * (2.0 * y[i - N2] + 2.0 * y[i - 2] -
                                     4.0 * y[i]);

      /* U(BRUSS_GRID_SIZE-1,j) */
      return 1.0
        + y[i] * y[i] * y[i + 1] - 4.4 * y[i]
        + BRUSS_ALPHA * N1 * N1
        * (2.0 * y[i - N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);
    }
    else
    {
      if (j == 0)
        /* U(k,0) */
        return 1.0
          + y[i] * y[i] * y[i + 1] - 4.4 * y[i]
          + BRUSS_ALPHA * N1 * N1
          * (y[i - N2] + y[i + N2] + 2.0 * y[i + 2] - 4.0 * y[i]);

      if (j == BRUSS_GRID_SIZE - 1)
        /* U(k,BRUSS_GRID_SIZE-1) */
        return 1.0
          + y[i] * y[i] * y[i + 1] - 4.4 * y[i]
          + BRUSS_ALPHA * N1 * N1
          * (y[i - N2] + y[i + N2] + 2.0 * y[i - 2] - 4.0 * y[i]);

      /* U(k,j) general */
      return 1.0
        + y[i] * y[i] * y[i + 1] - 4.4 * y[i]
        + BRUSS_ALPHA * N1 * N1
        * (y[i - N2] + y[i + N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);
    }
  }
  else                       /* --- V --------------------------------------- */
  {
    if (k == 0)
    {
      if (j == 0)
        /* V(0,0) */
        return 3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i]
          + BRUSS_ALPHA * N1 * N1 * (2.0 * y[i + N2] + 2.0 * y[i + 2] -
                                     4.0 * y[i]);

      if (j == BRUSS_GRID_SIZE - 1)
        /* V(0,BRUSS_GRID_SIZE-1) */
        return 3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i]
          + BRUSS_ALPHA * N1 * N1 * (2.0 * y[i + N2] + 2.0 * y[i - 2] -
                                     4.0 * y[i]);

      /* V(0, j) */
      return 3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i]
        + BRUSS_ALPHA * N1 * N1
        * (2.0 * y[i + N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);
    }
    else if (k == BRUSS_GRID_SIZE - 1)
    {
      if (j == 0)
        /* V(BRUSS_GRID_SIZE-1,0) */
        return 3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i]
          + BRUSS_ALPHA * N1 * N1 * (2.0 * y[i - N2] + 2.0 * y[i + 2] -
                                     4.0 * y[i]);

      if (j == BRUSS_GRID_SIZE - 1)
        /* V(BRUSS_GRID_SIZE-1,BRUSS_GRID_SIZE-1) */
        return 3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i]
          + BRUSS_ALPHA * N1 * N1 * (2.0 * y[i - N2] + 2.0 * y[i - 2] -
                                     4.0 * y[i]);

      /* V(BRUSS_GRID_SIZE-1,j) */
      return 3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i]
        + BRUSS_ALPHA * N1 * N1
        * (2.0 * y[i - N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);
    }
    else
    {
      if (j == 0)
        /* V(k,0) */
        return 3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i]
          + BRUSS_ALPHA * N1 * N1
          * (y[i - N2] + y[i + N2] + 2.0 * y[i + 2] - 4.0 * y[i]);

      if (j == BRUSS_GRID_SIZE - 1)
        /* V(k,BRUSS_GRID_SIZE-1) */
        return 3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i]
          + BRUSS_ALPHA * N1 * N1
          * (y[i - N2] + y[i + N2] + 2.0 * y[i - 2] - 4.0 * y[i]);

      /* V(k,j) general */
      return 3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i]
        + BRUSS_ALPHA * N1 * N1
        * (y[i - N2] + y[i + N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);
    }
  }
}

/******************************************************************************/
/* Access distance                                                            */
/******************************************************************************/

int ode_acc_dist()
{
  return 2 * BRUSS_GRID_SIZE;
}

/******************************************************************************/
