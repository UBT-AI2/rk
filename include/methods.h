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

#ifndef METHODS_H_
#define METHODS_H_

/******************************************************************************/

#include <math.h>
#include "tools.h"

/******************************************************************************/
/* Memory management                                                          */
/******************************************************************************/

static inline void alloc_emb_rk_method(double ***A, double **b, double **b_hat,
                                       double **c, int s)
{
  int i;
  *A = MALLOC(s, double *);
  (*A)[0] = MALLOC((s + 3) * s, double);
  for (i = 1; i < s; i++)
    (*A)[i] = (*A)[i - 1] + s;
  *b = (*A)[s - 1] + s;
  *b_hat = *b + s;
  *c = *b_hat + s;
}

static inline void free_emb_rk_method(double ***A, double **b, double **b_hat,
                                      double **c, int s)
{
  FREE((*A)[0]);
  FREE(*A);
  *b = *b_hat = *c = NULL;
}

static inline void alloc_zero_pattern(int ***A, int **b, int **b_hat,
                                      int **c, int s)
{
  int i;
  *A = MALLOC(s, int *);
  (*A)[0] = MALLOC((s + 3) * s, int);
  for (i = 1; i < s; i++)
    (*A)[i] = (*A)[i - 1] + s;
  *b = (*A)[s - 1] + s;
  *b_hat = *b + s;
  *c = *b_hat + s;
}

static inline void free_zero_pattern(int ***A, int **b, int **b_hat,
                                     int **c, int s)
{
  FREE((*A)[0]);
  FREE(*A);
  *b = *b_hat = *c = NULL;
}

/******************************************************************************/
/* Precompute flags for zero entries                                          */
/******************************************************************************/

void zero_pattern(double **A, double *b, double *b_hat, double *c,
                  int **is_zero_A, int *is_zero_b, int *is_zero_b_hat,
                  int *is_zero_c, int s);

/******************************************************************************/
/* Premultiplication with h                                                   */
/******************************************************************************/

void premult(double h, double **A, double *b, double *b_hat, double *c,
             double **hA, double *hb, double *hb_hat, double *hc, int s);

/******************************************************************************/
/* Embedded RK methods                                                        */
/******************************************************************************/

typedef void (emb_rk_method_t) (double ***A, double **b, double **b_hat,
                                double **c, int *s, int *ord);

emb_rk_method_t RKF23;          /* s =  3, ord = 2 */
emb_rk_method_t DOPRI54;        /* s =  7, ord = 5 */
emb_rk_method_t DOPRI87;        /* s = 13, ord = 8 */

/******************************************************************************/

#endif /* METHODS_H_ */

/******************************************************************************/
