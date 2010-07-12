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
#include "block-inline.h"

/******************************************************************************/

extern void solver(double t0, double te, double *y0, double *y, double tol);

/******************************************************************************/

#endif /* SOLVERS_H_ */

/******************************************************************************/
