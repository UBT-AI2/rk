/******************************************************************************/
/* This file is part of a collection of embedded Runge-Kutta solvers.         */
/* Copyright (C) 2009-2021, Matthias Korch, University of Bayreuth, Germany.  */
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

/******************************************************************************/
/* Test equation: LV                                                          */
/* Lotka-Volterra equations scaled by LV_COUNT                                */
/******************************************************************************/

#include "ode.h"
#include "solver.h"

/******************************************************************************/

int ode_size = 2 * LV_COUNT;

const double c0 = 0.1;
const double c1 = 0.2;
const double d0 = 0.2;
const double d1 = 0.4;

const double iv0 = 0.2;
const double iv1 = 1.1;

/******************************************************************************/
/* Initialization of the start vector (initial value)                         */
/******************************************************************************/

void ode_start(double t, double *y0)
{
  int i;

#ifdef HAVE_MPI
  if (me == 0)
  {
#endif
    printf("ODE: LV ");
    printf("(Lotka-Volterra, scaled)\n");
    printf("System size: %d\n", ode_size);
#ifdef HAVE_MPI
  }
#endif

  for (i = 0; i < ode_size; i += 2)
  {
    y0[i] = iv0;
    y0[i + 1] = iv1;
  }
}

/******************************************************************************/
/* Helper functions                                                           */
/******************************************************************************/

static inline double f0(double y0, double y1)
{
  return c0 * y0 - d0 * y0 * y1;        // prey
}

static inline double f1(double y0, double y1)
{
  return d1 * y0 * y1 - c1 * y1;
}

/******************************************************************************/
/* Evaluation of component i                                                  */
/******************************************************************************/

double ode_eval_comp(int i, double t, const double *y)
{
  return (i & 1) ? f1(y[i - 1], y[i]) : f0(y[i], y[i + 1]);

}

/******************************************************************************/
/* Evaluation of components i to j                                            */
/******************************************************************************/

void ode_eval_rng(int i, int j, double t, const double *y, double *f)
{
  if (j < i)
    return;

  if (i & 1)                    // i is odd (f1)
  {
    // evaluate f1 at i and move i to next even index
    f[i] = f1(y[i - 1], y[i]);
    i++;
  }

  while (i < j)
  {
    double y0 = y[i];
    double y1 = y[i + 1];

    f[i] = f0(y0, y1);
    f[i + 1] = f1(y0, y1);

    i += 2;
  }

  if (!(j & 1))                 // j is even (f0), so the while loop cannot have reached it
  {
    // evaluate f0 at j
    f[j] = f0(y[j], y[j + 1]);
  }
}

/******************************************************************************/
/* Access distance                                                            */
/******************************************************************************/

int ode_acc_dist()
{
  return 1;
}

/******************************************************************************/
