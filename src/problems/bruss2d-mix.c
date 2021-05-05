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

/******************************************************************************/
/* Test equation: BRUSS2D-MIX                                                 */
/* Brusselator function with ordering {U11, V11, U12, V12, ...}               */
/******************************************************************************/

#include "ode.h"
#include "solver.h"

/******************************************************************************/

int ode_size = 2 * BRUSS_GRID_SIZE * BRUSS_GRID_SIZE;

/******************************************************************************/
/* Initialization of the start vector (initial value)                         */
/******************************************************************************/

void ode_start(double t, double *y0)
{
  int i, j;

#ifdef HAVE_MPI
  if (me == 0)
  {
#endif
    printf("ODE: BRUSS2D-MIX ");
    printf("(2D Brusselator with mixed row-oriented ordering)\n");
    printf("Grid size: %d\n", BRUSS_GRID_SIZE);
    printf("Alpha: %.2e\n", BRUSS_ALPHA);
#ifdef HAVE_MPI
  }
#endif

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

  int k = i / N2;               /* row index */
  int j = (i - k * N2) / 2;     /* column index */
  int v = i & 1;                /* i even -> variable U; i odd -> variable V */

  if (!v)                       /* --- U ----------------------------------- */
  {
    if (k == 0)
    {
      if (j == 0)
      {
        /* U(0,0) */
        return 1.0
          + y[i] * y[i] * y[i + 1] - 4.4 * y[i]
          + BRUSS_ALPHA * N1 * N1 * (2.0 * y[i + N2] + 2.0 * y[i + 2] -
                                     4.0 * y[i]);
      }

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
  else                          /* --- V ----------------------------------- */
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
/* Evaluation of components i to j                                            */
/******************************************************************************/

static inline void eval_top_row(int i, int j, const double *y, double *f,
                                /* arguments to pass precalculated values */
                                int N, double N1, int N2, double alpha)
{
  /* right corner included? */
  const int right_u = N2 - 2;
  if (j >= right_u)
  {
    const int right_v = N2 - 1;

    /* j is on the corner
       so the only way U could NOT be included
       is if i == j == V */
    if (i != right_v)
    {
      /* U(0,N-1) */
      f[right_u] =
        1.0 + y[right_u] * y[right_u] * y[right_u + 1] - 4.4 * y[right_u] +
        alpha * N1 * N1 *
        (2.0 * y[right_u + N2] + 2.0 * y[right_u - 2] - 4.0 * y[right_u]);
    }

    if (j == right_v)
    {
      /* V(0,N-1) */
      f[right_v] =
        3.4 * y[right_v - 1] - y[right_v - 1] * y[right_v - 1] * y[right_v] +
        alpha * N1 * N1 *
        (2.0 * y[right_v + N2] + 2.0 * y[right_v - 2] - 4.0 * y[right_v]);
    }

    /* the range only covers the corner */
    if (i >= right_u)
    {
      return;
    }

    /* corner handled */
    j = N2 - 3;
  }

  /* left corner included? */
  if (i < 2)
  {
    /* i == 0 */
    if (!i)
    {
      /* U(0,0) */
      f[0] = 1.0 + y[0] * y[0] * y[1] - 4.4 * y[0] +
        alpha * N1 * N1 * (2.0 * y[N2] + 2.0 * y[2] - 4.0 * y[0]);
    }

    /* j != 0 */
    if (j)
    {
      /* V(0,0) */
      f[1] = 3.4 * y[0] - y[0] * y[0] * y[1] +
        alpha * N1 * N1 * (2.0 * y[1 + N2] + 2.0 * y[1 + 2] - 4.0 * y[1]);
    }

    if (j < 2)
    {
      return;
    }

    /* corner case handled */
    i = 2;
  }

  /* range starts on a v */
  if (i & 1)
  {
    /* V(0, j) */
    f[i] =
      3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i] +
      alpha * N1 * N1 * (2.0 * y[i + N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);

    if (i == j)
    {
      return;
    }

    ++i;
  }

  /* range ends on a u */
  if (!(j & 1))
  {
    /* U(0, j) */
    f[j] =
      1.0 + y[j] * y[j] * y[j + 1] - 4.4 * y[j] +
      alpha * N1 * N1 * (2.0 * y[j + N2] + y[j - 2] + y[j + 2] - 4.0 * y[j]);

    if (i == j)
    {
      return;
    }

    --j;
  }

  /* handle non-corner cells
     in pairs of (u,v) */
  for (; i < j; i += 2)
  {
    /* U(0, j) */
    f[i] =
      1.0 + y[i] * y[i] * y[i + 1] - 4.4 * y[i] +
      alpha * N1 * N1 * (2.0 * y[i + N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);
    /* V(0, j) */
    f[i + 1] = 3.4 * y[i] - y[i] * y[i] * y[i + 1] +
      alpha * N1 * N1 *
      (2.0 * y[i + 1 + N2] + y[i - 1] + y[i + 3] - 4.0 * y[i + 1]);
  }
}

static inline void eval_left_column(int i, int j, const double *y, double *f,
                                    // precalculated values
                                    int N, double N1, int N2, double alpha)
{
  const int imodN2 = i % N2;
  const int jmodN2 = j % N2;

  /* first edge cell is on a v */
  if (imodN2 == 1)
  {
    /* V(k,0) */
    f[i] =
      3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i] +
      alpha * N1 * N1 * (y[i - N2] + y[i + N2] + 2.0 * y[i + 2] - 4.0 * y[i]);

    if (i == j)
    {
      return;
    }
  }

  if (imodN2 > 0)
  {
    /* go to the beginning of the first edge pair */
    i = i + N2 - imodN2;
  }

  /* last edge cell is on a u */
  if (jmodN2 == 0)
  {
    /* U(k,0) */
    f[j] =
      1.0 + y[j] * y[j] * y[j + 1] - 4.4 * y[j] +
      alpha * N1 * N1 * (y[j - N2] + y[j + N2] + 2.0 * y[j + 2] - 4.0 * y[j]);

    if (i == j)
    {
      return;
    }

    --j;
  }

  if (j < i)
  {
    /* we moved i out of the range */
    return;
  }

  /* handle edge cells in pairs */
  for (; i < j; i += N2)
  {
    /* U(k,0) */
    f[i] =
      1.0 + y[i] * y[i] * y[i + 1] - 4.4 * y[i] +
      alpha * N1 * N1 * (y[i - N2] + y[i + N2] + 2.0 * y[i + 2] - 4.0 * y[i]);
    /* V(k,0) */
    f[i + 1] = 3.4 * y[i] - y[i] * y[i] * y[i + 1] +
      alpha * N1 * N1 * (y[i + 1 - N2] + y[i + 1 + N2] +
                         2.0 * y[i + 3] - 4.0 * y[i + 1]);
  }
}

static inline void eval_right_column(int i, int j, const double *y, double *f,
                                     /* precalculated values */
                                     int N, double N1, int N2, double alpha)
{
  const int imodN2 = i % N2;
  const int jmodN2 = j % N2;

  /* first edge cell is on a v */
  if (imodN2 == N2 - 1)
  {
    /* V(k,N-1) */
    f[i] =
      3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i] +
      alpha * N1 * N1 * (y[i - N2] + y[i + N2] + 2.0 * y[i - 2] - 4.0 * y[i]);

    if (i == j)
    {
      return;
    }

    /* go to the next edge pair */
    i = i + N2 - 1;
  }
  else if (imodN2 < N2 - 2)
  {
    /* go to the beginning of the first edge pair */
    i = i + N2 - imodN2 - 2;
  }

  /* last edge cell is on a u */
  if (jmodN2 == N2 - 2)
  {
    /* U(k,N-1) */
    f[j] =
      1.0 + y[j] * y[j] * y[j + 1] - 4.4 * y[j] +
      alpha * N1 * N1 * (y[j - N2] + y[j + N2] + 2.0 * y[j - 2] - 4.0 * y[j]);

    if (i == j)
    {
      return;
    }

    --j;
  }

  if (j < i)
  {
    return;
  }

  /* handle edge cells in pairs */
  for (; i < j; i += N2)
  {
    /* U(k,N-1) */
    f[i] =
      1.0 + y[i] * y[i] * y[i + 1] - 4.4 * y[i] +
      alpha * N1 * N1 * (y[i - N2] + y[i + N2] + 2.0 * y[i - 2] - 4.0 * y[i]);
    /* V(k,N-1) */
    f[i + 1] = 3.4 * y[i] - y[i] * y[i] * y[i + 1] +
      alpha * N1 * N1 * (y[i + 1 - N2] + y[i + 1 + N2] +
                         2.0 * y[i - 1] - 4.0 * y[i + 1]);
  }
}

static inline void eval_bottom_row(int i, int j, const double *y, double *f,
                                   /* arguments to pass precalculated values */
                                   int N, double N1, int N2, double alpha,
                                   int NTotal)
{
  /* right corner included? */
  const int right_u = NTotal - 2;
  if (j >= right_u)
  {
    const int right_v = right_u + 1;

    /* j is on the corner
       so the only way U could NOT be included
       is if i == j == V */
    if (i != right_v)
    {
      /* U(N-1,N-1) */
      f[right_u] =
        1.0 + y[right_u] * y[right_u] * y[right_u + 1] - 4.4 * y[right_u] +
        alpha * N1 * N1 *
        (2.0 * y[right_u - N2] + 2.0 * y[right_u - 2] - 4.0 * y[right_u]);
    }

    if (j == right_v)
    {
      /* V(N-1,N-1) */
      f[right_v] = 3.4 * y[right_u] - y[right_u] * y[right_u] * y[right_v] +
        alpha * N1 * N1 * (2.0 * y[right_v - N2] +
                           2.0 * y[right_v - 2] - 4.0 * y[right_v]);
    }

    /* the range only covers the corner */
    if (i >= right_u)
    {
      return;
    }

    /* corner handled */
    j = NTotal - 3;
  }

  /* left corner included? */
  const int left_v = NTotal - N2 + 1;
  if (i <= left_v)
  {
    const int left_u = left_v - 1;

    if (i == left_u)
    {
      /* U(N-1,0) */
      f[left_u] =
        1.0 + y[left_u] * y[left_u] * y[left_u + 1] - 4.4 * y[left_u] +
        alpha * N1 * N1 *
        (2.0 * y[left_u - N2] + 2.0 * y[left_u + 2] - 4.0 * y[left_u]);
    }

    if (j != left_u)
    {
      /* V(N-1,0) */
      f[left_v] = 3.4 * y[left_u] - y[left_u] * y[left_u] * y[left_v] +
        alpha * N1 * N1 * (2.0 * y[left_v - N2] +
                           2.0 * y[left_v + 2] - 4.0 * y[left_v]);
    }

    if (j <= left_v)
    {
      return;
    }

    /* corner case handled */
    i = left_v + 1;
  }

  /* range starts on a v */
  if (i & 1)
  {
    /* V(N-1,j) */
    f[i] =
      3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i] +
      alpha * N1 * N1 * (2.0 * y[i - N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);

    if (i == j)
    {
      return;
    }

    ++i;
  }

  /* range ends on a u */
  if (!(j & 1))
  {
    /* U(N-1,j) */
    f[j] =
      1.0 + y[j] * y[j] * y[j + 1] - 4.4 * y[j] +
      alpha * N1 * N1 * (2.0 * y[j - N2] + y[j - 2] + y[j + 2] - 4.0 * y[j]);

    if (i == j)
    {
      return;
    }

    --j;
  }

  /* handle non-corner cells
     in pairs of (u,v) */
  for (; i < j; i += 2)
  {
    /* U(N-1,j) */
    f[i] =
      1.0 + y[i] * y[i] * y[i + 1] - 4.4 * y[i] +
      alpha * N1 * N1 * (2.0 * y[i - N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);
    /* V(N-1,j) */
    f[i + 1] = 3.4 * y[i] - y[i] * y[i] * y[i + 1] +
      alpha * N1 * N1 *
      (2.0 * y[i + 1 - N2] + y[i - 1] + y[i + 3] - 4.0 * y[i + 1]);
  }
}

/* calculates the offset to the next larger
   cell index that is not an edge cell
   (edge cells are those with i % N2 ==
   N2 - 2, N2 - 1, 0 or 1) */
static inline int non_edge_ceil_offset(int i, int N2)
{
  int offset_mod = (i + 2) % N2;
  int diff = 4 - imin(4, offset_mod);

  return diff;
}

static inline int non_edge_ceil(int i, int N2)
{
  return i + non_edge_ceil_offset(i, N2);
}

static inline int non_edge_floor(int i, int N2)
{
  int ceil_offset = non_edge_ceil_offset(i, N2);
  if (ceil_offset)
  {
    return i - (5 - ceil_offset);
  }
  else
  {
    return i;
  }
}

static inline void eval_inner(int i, int j, const double *y, double *f,
                              /* precomputed values */
                              int N, double N1, int N2, double alpha)
{
  /* move borders out of the edge columns, if necessary */
  i = non_edge_ceil(i, N2);
  j = non_edge_floor(j, N2);
  int imodN2 = i % N2;
  int start = i - imodN2 + 2;
  int end = start + N2 - 5;

  if (j < i)
  {
    return;
  }

  /* first cell is on a v */
  if (i & 1)
  {
    /* V(k,j) */
    f[i] = 3.4 * y[i - 1] - y[i - 1] * y[i - 1] * y[i] +
      alpha * N1 * N1 *
      (y[i - N2] + y[i + N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);

    if (i == j)
    {
      return;
    }

    ++i;
  }

  /* last edge cell is on a u */
  if (!(j & 1))
  {
    /* U(k,j) */
    f[j] = 1.0 + y[j] * y[j] * y[j + 1] - 4.4 * y[j] +
      alpha * N1 * N1 *
      (y[j - N2] + y[j + N2] + y[j - 2] + y[j + 2] - 4.0 * y[j]);

    if (i == j)
    {
      return;
    }

    --j;
  }

  for (; i < j; i = (start += N2), end += N2)
  {
    int k = imin(end, j);
    for (; i < k; i += 2)
    {
      /* U(k,j) */
      f[i] = 1.0 + y[i] * y[i] * y[i + 1] - 4.4 * y[i] +
        alpha * N1 * N1 *
        (y[i - N2] + y[i + N2] + y[i - 2] + y[i + 2] - 4.0 * y[i]);
      /* V(k,j) */
      f[i + 1] = 3.4 * y[i] - y[i] * y[i] * y[i + 1] +
        alpha * N1 * N1 * (y[i + 1 - N2] + y[i + 1 + N2] + y[i - 1] +
                           y[i + 3] - 4.0 * y[i + 1]);
    }
  }
}

void ode_eval_rng(int i, int j, double t, const double *y, double *f)
{
  const double alpha = BRUSS_ALPHA;
  const int N = BRUSS_GRID_SIZE;
  const double N1 = (double) N - 1.0;
  /* row size */
  const int N2 = N + N;
  /* total grid size */
  const int NTotal = N * N2;
  /* rows for i and j */
  const int ri = i / N2;
  const int rj = j / N2;

  /* top row components, if any */
  if (ri == 0)
  {
    /* last component on the top row */
    const int tj = imin(N2 - 1, j);

    eval_top_row(i, tj, y, f, N, N1, N2, alpha);
  }

  // inner row components, if any
  if (i < (NTotal - N2) && j >= N2)
  {
    const int ii = imax(N2, i);
    const int ij = imin(NTotal - N2 - 1, j);

    eval_left_column(ii, ij, y, f, N, N1, N2, alpha);

    eval_inner(ii, ij, y, f, N, N1, N2, alpha);

    eval_right_column(ii, ij, y, f, N, N1, N2, alpha);
  }

  // bottom row components, if any
  if (rj >= N - 1)
  {
    const int bi = imax(NTotal - N2, i);

    eval_bottom_row(bi, j, y, f, N, N1, N2, alpha, NTotal);
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
