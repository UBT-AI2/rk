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

#ifndef PIPE_INLINE_H_
#define PIPE_INLINE_H_

/******************************************************************************/
#include "config.h"
#include "tools.h"
#include "ode.h"

/******************************************************************************/

#define BKSTAGE0(p0) \
  bkstage0(p0, s, t, h, A, b, bbs, c, y, err_vec, delta_y, w)

static inline void bkstage0(int p0, int s, double t,
			    double h, double **A, double *b,
			    double *bbs, double *c, double *y,
			    double *err_vec, double *delta_y,
			    double **w)
{                                                                     
  int p, q, pe = p0 + BLOCKSIZE;                                            

  for (p = p0; p < pe; p++)                                          
  {                                                                  
    double F = h * ode_eval_comp(p, t + c[0] * h, y);             
    double Y = y[p];
    
    for (q = 1; q < s; q++) 
      w[q][p] = Y + A[q][0] * F;       
    
    delta_y[p] = b[0] * F; 
    err_vec[p] = bbs[0] * F;                 
  }                                                                  
}

/******************************************************************************/

#define BKSTAGEn(p0, m) \
  bkstagen(p0, m, s, t, h, A, b, bbs, c, y, err_vec, delta_y, w)

static inline void bkstagen(int p0, int m, int s,
			    double t, double h, double **A, double *b,
			    double *bbs, double *c, double *y,
			    double *err_vec, double *delta_y,
			    double **w)
{                                                                    
  int p, q, pe = p0 + BLOCKSIZE;                                            

  for (p = p0; p < pe; p++)                                           
  {                                                                   
    double F = h * ode_eval_comp(p, t + c[m] * h, w[m]);      

    for (q = (m)+1; q < s; q++) 
      w[q][p] += A[q][m] * F;       

    delta_y[p] += b[m] * F; 
    err_vec[p] += bbs[m] * F;                
  }                                                                   
}

/******************************************************************************/

#define BKSTAGEsm1(p0) \
  bkstagesm1(p0, s, t, h, b, bbs, c, y, err_vec, delta_y, w, &error_max)

static inline void bkstagesm1(int p0, int s,
			      double t, double h, double *b,
			      double *bbs, double *c, double *y,
			      double *err_vec, double *delta_y,
			      double **w, double *error_max)
{                                                                    
  int p, pe = p0 + BLOCKSIZE, m = s - 1;

  for (p = p0; p < pe; p++)                                           
  {                                                          
    double yp_old;
    double F = h * ode_eval_comp(p, t + c[m] * h, w[m]);     

    delta_y[p] += b[m] * F; 
    err_vec[p] += bbs[m] * F;                

    yp_old = y[p];
    y[p] += h * delta_y[p];
    delta_y[p] = yp_old;        /* y_old and delta_y occupy the same space */

    update_error_max(error_max, err_vec[p], y[p], yp_old);
  }                                                                   
}

/******************************************************************************/

#endif /* PIPE_INLINE */

/******************************************************************************/
