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

#ifndef CONFIG_H_
#define CONFIG_H_

/******************************************************************************/
/* User defined options                                                       */
/******************************************************************************/

/*
 * Parameters of the Brusselator equation
 */

#ifndef BRUSS_GRID_SIZE
#define BRUSS_GRID_SIZE         250
#endif

#ifndef BRUSS_ALPHA
#define BRUSS_ALPHA             2E-3
#endif

/*
 * Integration interval
 */

#ifndef T_START
#define T_START                 0.0
#endif

#ifndef T_END
#define T_END                   0.5
#endif

/*
 * Tolerance of the stepsize controller
 */

#ifndef TOL
#define TOL                     1E-9
#endif

/*
 * Maximum number of steps
 */

#ifndef STEP_LIMIT
#define STEP_LIMIT              200
#endif

/*
 * Available Methods:
 * 
 *   RKF23
 *   DOPRI54
 *   DOPRI87
 */

#ifndef METHOD
#define METHOD                  DOPRI54
#endif

/*
 * Blocksize to be used by the pipelining implementations. Must be
 * larger than or equal to 2 * BRUSS_GRID_SIZE.
 */

#ifndef BLOCKSIZE
#define BLOCKSIZE               (2 * (BRUSS_GRID_SIZE))
#endif

/*
 * Type of locks to be used (SPINLOCK or MUTEX)
 */

#ifndef LOCKTYPE
#define LOCKTYPE                MUTEX
#endif

/*
 * Number of bytes used for the padding of data structures 
 */

#ifndef PAD_SIZE
#define PAD_SIZE                128
#endif

/*
 * Use memcpy or a 'for' loop to implement the vector copy operation.
 */

#define USE_MEMCPY

/******************************************************************************/

#endif /* CONFIG_H_ */

/******************************************************************************/
