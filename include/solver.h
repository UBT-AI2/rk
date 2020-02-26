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

#ifndef SOLVERS_H_
#define SOLVERS_H_

/******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "config.h"
#include "ode.h"
#include "methods.h"
#include "threads.h"
#include "tools.h"

/******************************************************************************/

#ifdef HAVE_MPI
#include <mpi.h>
extern int me;
extern int processes;
#endif

extern int threads;

/******************************************************************************/

extern void solver(double t0, double te, double *y0, double *y, double tol);

/******************************************************************************/

#include "block-inline.h"       /* must be included _after_ the declaration of
                                   me, processes, and threads */

/******************************************************************************/

#endif /* SOLVERS_H_ */

/******************************************************************************/
