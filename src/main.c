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

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "config.h"
#include "ode.h"
#include "solver.h"
#include "tools.h"

/******************************************************************************/

#ifdef HAVE_MPI
#include <mpi.h>
int me;
int processes;
#endif

int threads;

/******************************************************************************/

int main(int argc, char *argv[])
{
  double *y_0, *y;
  double min, max;
  int i;
  double timer;
  char *env_threads;
  double tol = HAS_EMB_SOL(METHOD) ? TOL : 0.0;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, (int *) &me);
  MPI_Comm_size(MPI_COMM_WORLD, (int *) &processes);
  assert(processes <= ode_size);
#endif

  env_threads = getenv("NUM_THREADS");
  threads =
    ((env_threads != NULL) ? imin(imax(1, atoi(env_threads)), ode_size) : 1);

  y_0 = MALLOC(ode_size, double);
  y = MALLOC(ode_size, double);

  ode_start(T_START, y_0);

#ifdef HAVE_MPI
  if (me == 0)
  {
#endif
    for (min = max = y_0[0], i = 1; i < ode_size; i++)
      if (y_0[i] < min)
        min = y_0[i];
      else if (y_0[i] > max)
        max = y_0[i];

    printf("Integration interval: [%.2e,%.2e]\n", T_START, T_END);
    printf("Initial value: y0[0] = %e, ..., y0[%d] = %e; min = %e, max = %e\n",
           y_0[0], ode_size - 1, y_0[ode_size - 1], min, max);
    printf("Tolerance: %.2e\n", tol);

#if LOCKTYPE == SPINLOCK
    printf("Lock type: spin\n");
#else
    printf("Lock type: mutex\n");
#endif

    printf("Pad size: %d\n", PAD_SIZE);
#ifdef HAVE_MPI
  }
#endif

  timer_start(&timer);
  solver(T_START, T_END, y_0, y, tol);
  timer_stop(&timer);

#ifdef HAVE_MPI
  if (me == 0)
  {
#endif
    printf("Total time:  %.2e s\n", timer);

    for (min = max = y[0], i = 1; i < ode_size; i++)
      if (y[i] < min)
        min = y[i];
      else if (y[i] > max)
        max = y[i];

    printf
      ("Result: y[0] = %.20e, ..., y[%d] = %.20e; min = %.20e, max = %.20e\n",
       y[0], ode_size - 1, y[ode_size - 1], min, max);
#ifdef HAVE_MPI
  }
#endif

  FREE(y_0);
  FREE(y);

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  exit(0);
}

/******************************************************************************/
