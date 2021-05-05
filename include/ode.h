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

#ifndef ODE_H_
#define ODE_H_

/******************************************************************************/

#include "config.h"
#include <stdio.h>

/******************************************************************************/

extern int ode_size;

/******************************************************************************/

extern void ode_start(double t, double *y0);
extern double ode_eval_comp(int i, double t, const double *y);
extern void ode_eval_rng(int i, int j, double t, const double *y, double *f);
extern int ode_acc_dist(void);

/******************************************************************************/

#endif /* ODE_H_ */

/******************************************************************************/
