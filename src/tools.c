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

#include "solver.h"

/******************************************************************************/
/* Selection of the initial stepsize                                          */
/******************************************************************************/

double initial_stepsize(double t0, double H, const double *y_0, int ord,
                        double tol)
{
  double h;
#ifdef H_INIT
  h = H_INIT;
#else
  h = tol > 0.0 ? fmin(10.0 * tol, H / (STEP_LIMIT)) : H / (STEP_LIMIT);
#endif
  printf("Initial step size: %.e\n", h);
  return h;
}

/******************************************************************************/
/* Data distribution                                                          */
/******************************************************************************/

void blockwise_distribution(int processors, int n, int *first, int *size)
{
  int i;

  assert(n >= processors);

  int q = n / processors;
  int r = n % processors;

  for (i = 0; i < processors; i++)
  {
    first[i] = i < r ? i * (q + 1) : r * (q + 1) + (i - r) * q;
    size[i] = imin(i < r ? q + 1 : q, n - first[i]);
  }
}

/******************************************************************************/
/* Print statistics                                                           */
/******************************************************************************/

void print_statistics(double timer, int steps_acc, int steps_rej)
{
#ifdef HAVE_MPI
  if (me > 0)
    return;
#endif

  printf("Number of steps: %d (%d accepted, %d rejected)\n",
         steps_acc + steps_rej, steps_acc, steps_rej);

#if (STEP_LIMIT > 0)
  if (steps_acc + steps_rej >= STEP_LIMIT)
    printf("Step limit reached.\n");
#endif

  printf("Kernel time: %.2e s (%.2e s per step)\n", timer,
         timer / (double) (steps_acc + steps_rej));
}

/******************************************************************************/
