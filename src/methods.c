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

#include "methods.h"
#include "solver.h"

/******************************************************************************/

void zero_pattern(double **A, double *b, double *b_hat, double *c,
                  int **is_zero_A, int *is_zero_b, int *is_zero_b_hat,
                  int *is_zero_c, int s)
{
  int i, j;

  for (i = 0; i < s; i++)
  {
    for (j = 0; j < s; j++)
      is_zero_A[i][j] = (A[i][j] != 0.0);

    is_zero_b[i] = (b[i] != 0.0);
    is_zero_b_hat[i] = (b_hat[i] != 0.0);
    is_zero_c[i] = (c[i] != 0.0);
  }
}

/******************************************************************************/

void premult(double h, double **A, double *b, double *b_hat, double *c,
             double **hA, double *hb, double *hb_hat, double *hc, int s)
{
  int i, j;

  for (i = 0; i < s; i++)
  {

    for (j = 0; j < i; j++)
      hA[i][j] = h * A[i][j];

    hb[i] = h * b[i];
    hb_hat[i] = h * b_hat[i];
    hc[i] = h * c[i];
  }
}

/******************************************************************************/

void RKF23(double ***A, double **b, double **b_hat, double **c, int *s,
           int *ord)
{
  /*
   * Source: K. Strehmel, R. Weiner, Numerik gewöhnlicher
   * Differentialgleichungen, Teubner 1995, p. 64
   */

  *s = 3;
  *ord = 2;

#ifdef HAVE_MPI
  if (me == 0)
  {
#endif
    printf("Method: RKF 2(3) (order %d, %d stages)\n", *ord, *s);
#ifdef HAVE_MPI
  }
#endif

  alloc_emb_rk_method(A, b, b_hat, c, *s);

  (*c)[0] = 0.0;
  (*c)[1] = 1.0;
  (*c)[2] = 1.0 / 2.0;

  (*b)[0] = 1.0 / 2.0;
  (*b)[1] = 1.0 / 2.0;
  (*b)[2] = 0.0;

  (*b_hat)[0] = 1.0 / 6.0;
  (*b_hat)[1] = 1.0 / 6.0;
  (*b_hat)[2] = 4.0 / 6.0;

  (*A)[0][0] = 0.0;
  (*A)[1][0] = 1.0;
  (*A)[2][0] = 1.0 / 4.0;

  (*A)[0][1] = 0.0;
  (*A)[1][1] = 0.0;
  (*A)[2][1] = 1.0 / 4.0;

  (*A)[0][2] = 0.0;
  (*A)[1][2] = 0.0;
  (*A)[2][2] = 0.0;
}


/******************************************************************************/

void DOPRI54(double ***A, double **b, double **b_hat, double **c, int *s,
             int *ord)
{
  /*
   * Sources:
   * - E. Hairer, S.P. Norsett, G. Wanner, Solving Ordinary Differential
   *   Equations I -- Nonstiff Problems, 2nd rev. ed., Springer 2000, p. 178
   * - J.C. Butcher, Numerical Methods for Ordinary Differential Equations,
   *   2nd. ed., Wiley 2008, p. 211
   */

  *s = 7;
  *ord = 5;

#ifdef HAVE_MPI
  if (me == 0)
  {
#endif
    printf("Method: DOPRI 5(4) (order %d, %d stages)\n", *ord, *s);
#ifdef HAVE_MPI
  }
#endif

  alloc_emb_rk_method(A, b, b_hat, c, *s);

  (*c)[0] = 0.0;
  (*c)[1] = 1.0 / 5.0;
  (*c)[2] = 3.0 / 10.0;
  (*c)[3] = 4.0 / 5.0;
  (*c)[4] = 8.0 / 9.0;
  (*c)[5] = 1.0;
  (*c)[6] = 1.0;

  (*b)[0] = 35.0 / 384.0;
  (*b)[1] = 0.0;
  (*b)[2] = 500.0 / 1113.0;
  (*b)[3] = 125.0 / 192.0;
  (*b)[4] = -2187.0 / 6784.0;
  (*b)[5] = 11.0 / 84.0;
  (*b)[6] = 0.0;

  (*b_hat)[0] = 5179.0 / 57600;
  (*b_hat)[1] = 0.0;
  (*b_hat)[2] = 7571.0 / 16695;
  (*b_hat)[3] = 393.0 / 640.0;
  (*b_hat)[4] = -92097.0 / 339200;
  (*b_hat)[5] = 187.0 / 2100.0;
  (*b_hat)[6] = 1.0 / 40.0;

  (*A)[0][0] = 0.0;
  (*A)[0][1] = 0.0;
  (*A)[0][2] = 0.0;
  (*A)[0][3] = 0.0;
  (*A)[0][4] = 0.0;
  (*A)[0][5] = 0.0;
  (*A)[0][6] = 0.0;

  (*A)[1][0] = 1.0 / 5.0;
  (*A)[1][1] = 0.0;
  (*A)[1][2] = 0.0;
  (*A)[1][3] = 0.0;
  (*A)[1][4] = 0.0;
  (*A)[1][5] = 0.0;
  (*A)[1][6] = 0.0;

  (*A)[2][0] = 3.0 / 40.0;
  (*A)[2][1] = 9.0 / 40.0;
  (*A)[2][2] = 0.0;
  (*A)[2][3] = 0.0;
  (*A)[2][4] = 0.0;
  (*A)[2][5] = 0.0;
  (*A)[2][6] = 0.0;

  (*A)[3][0] = 44.0 / 45.0;
  (*A)[3][1] = -56.0 / 15.0;
  (*A)[3][2] = 32.0 / 9.0;
  (*A)[3][3] = 0.0;
  (*A)[3][4] = 0.0;
  (*A)[3][5] = 0.0;
  (*A)[3][6] = 0.0;

  (*A)[4][0] = 19372.0 / 6561.0;
  (*A)[4][1] = -25360.0 / 2187.0;
  (*A)[4][2] = 64448.0 / 6561.0;
  (*A)[4][3] = -212.0 / 729.0;
  (*A)[4][4] = 0.0;
  (*A)[4][5] = 0.0;
  (*A)[4][6] = 0.0;

  (*A)[5][0] = 9017.0 / 3168.0;
  (*A)[5][1] = -355.0 / 33.0;
  (*A)[5][2] = 46732.0 / 5247.0;
  (*A)[5][3] = 49.0 / 176.0;
  (*A)[5][4] = -5103.0 / 18656.0;
  (*A)[5][5] = 0.0;
  (*A)[5][6] = 0.0;

  (*A)[6][0] = 35.0 / 384.0;
  (*A)[6][1] = 0.0;
  (*A)[6][2] = 500.0 / 1113.0;
  (*A)[6][3] = 125.0 / 192.0;
  (*A)[6][4] = -2187.0 / 6784.0;
  (*A)[6][5] = 11.0 / 84.0;
  (*A)[6][6] = 0.0;
}

/******************************************************************************/

void DOPRI87(double ***A, double **b, double **b_hat, double **c, int *s,
             int *ord)
{
  /*
   * Sources:
   * - P.J. Prince, J.R. Dormand, High Order Embedded Runge-Kutta Formulae,
   *   J. Comp. Appl. Math. 7 (1981), p. 67-75
   * - E. Hairer, S.P. Norsett, G. Wanner, Solving Ordinary Differential
   *   Equations I -- Nonstiff Problems, 1st ed., Springer 1987, sec. II.6
   */

  *s = 13;
  *ord = 8;

#ifdef HAVE_MPI
  if (me == 0)
  {
#endif
    printf("Method: DOPRI 8(7) (order %d, %d stages)\n", *ord, *s);
#ifdef HAVE_MPI
  }
#endif

  alloc_emb_rk_method(A, b, b_hat, c, *s);

  (*c)[0] = 0.0;
  (*c)[1] = 1.0 / 18.0;
  (*c)[2] = 1.0 / 12.0;
  (*c)[3] = 1.0 / 8.0;
  (*c)[4] = 5.0 / 16.0;
  (*c)[5] = 3.0 / 8.0;
  (*c)[6] = 59.0 / 400.0;
  (*c)[7] = 93.0 / 200.0;
  (*c)[8] = 5490023248.0 / 9719169821.0;
  (*c)[9] = 13.0 / 20.0;
  (*c)[10] = 1201146811.0 / 1299019798.0;
  (*c)[11] = 1.0;
  (*c)[12] = 1.0;

  (*b)[0] = 14005451.0 / 335480064.0;
  (*b)[1] = 0.0;
  (*b)[2] = 0.0;
  (*b)[3] = 0.0;
  (*b)[4] = 0.0;
  (*b)[5] = -59238493.0 / 1068277825.0;
  (*b)[6] = 181606767.0 / 758867731.0;
  (*b)[7] = 561292985.0 / 797845732.0;
  (*b)[8] = -1041891430.0 / 1371343529.0;
  (*b)[9] = 760417239.0 / 1151165299.0;
  (*b)[10] = 118820643.0 / 751138087.0;
  (*b)[11] = -528747749.0 / 2220607170.0;
  (*b)[12] = 1.0 / 4.0;

  (*b_hat)[0] = 13451932.0 / 455176623.0;
  (*b_hat)[1] = 0.0;
  (*b_hat)[2] = 0.0;
  (*b_hat)[3] = 0.0;
  (*b_hat)[4] = 0.0;
  (*b_hat)[5] = -808719846.0 / 976000145.0;
  (*b_hat)[6] = 1757004468.0 / 5645159321.0;
  (*b_hat)[7] = 656045339.0 / 265891186.0;
  (*b_hat)[8] = -3867574721.0 / 1518517206.0;
  (*b_hat)[9] = 465885868.0 / 322736535.0;
  (*b_hat)[10] = 53011238.0 / 667516719.0;
  (*b_hat)[11] = 2.0 / 45.0;
  (*b_hat)[12] = 0.0;

  (*A)[0][0] = 0.0;
  (*A)[0][1] = 0.0;
  (*A)[0][2] = 0.0;
  (*A)[0][3] = 0.0;
  (*A)[0][4] = 0.0;
  (*A)[0][5] = 0.0;
  (*A)[0][6] = 0.0;
  (*A)[0][7] = 0.0;
  (*A)[0][8] = 0.0;
  (*A)[0][9] = 0.0;
  (*A)[0][10] = 0.0;
  (*A)[0][11] = 0.0;
  (*A)[0][12] = 0.0;

  (*A)[1][0] = 1.0 / 18.0;
  (*A)[1][1] = 0.0;
  (*A)[1][2] = 0.0;
  (*A)[1][3] = 0.0;
  (*A)[1][4] = 0.0;
  (*A)[1][5] = 0.0;
  (*A)[1][6] = 0.0;
  (*A)[1][7] = 0.0;
  (*A)[1][8] = 0.0;
  (*A)[1][9] = 0.0;
  (*A)[1][10] = 0.0;
  (*A)[1][11] = 0.0;
  (*A)[1][12] = 0.0;

  (*A)[2][0] = 1.0 / 48.0;
  (*A)[2][1] = 1.0 / 16.0;
  (*A)[2][2] = 0.0;
  (*A)[2][3] = 0.0;
  (*A)[2][4] = 0.0;
  (*A)[2][5] = 0.0;
  (*A)[2][6] = 0.0;
  (*A)[2][7] = 0.0;
  (*A)[2][8] = 0.0;
  (*A)[2][9] = 0.0;
  (*A)[2][10] = 0.0;
  (*A)[2][11] = 0.0;
  (*A)[2][12] = 0.0;

  (*A)[3][0] = 1.0 / 32.0;
  (*A)[3][1] = 0.0;
  (*A)[3][2] = 3.0 / 32.0;
  (*A)[3][3] = 0.0;
  (*A)[3][4] = 0.0;
  (*A)[3][5] = 0.0;
  (*A)[3][6] = 0.0;
  (*A)[3][7] = 0.0;
  (*A)[3][8] = 0.0;
  (*A)[3][9] = 0.0;
  (*A)[3][10] = 0.0;
  (*A)[3][11] = 0.0;
  (*A)[3][12] = 0.0;

  (*A)[4][0] = 5.0 / 16.0;
  (*A)[4][1] = 0.0;
  (*A)[4][2] = -75.0 / 64.0;
  (*A)[4][3] = 75.0 / 64.0;
  (*A)[4][4] = 0.0;
  (*A)[4][5] = 0.0;
  (*A)[4][6] = 0.0;
  (*A)[4][7] = 0.0;
  (*A)[4][8] = 0.0;
  (*A)[4][9] = 0.0;
  (*A)[4][10] = 0.0;
  (*A)[4][11] = 0.0;
  (*A)[4][12] = 0.0;

  (*A)[5][0] = 3.0 / 80.0;
  (*A)[5][1] = 0.0;
  (*A)[5][2] = 0.0;
  (*A)[5][3] = 3.0 / 16.0;
  (*A)[5][4] = 3.0 / 20.0;
  (*A)[5][5] = 0.0;
  (*A)[5][6] = 0.0;
  (*A)[5][7] = 0.0;
  (*A)[5][8] = 0.0;
  (*A)[5][9] = 0.0;
  (*A)[5][10] = 0.0;
  (*A)[5][11] = 0.0;
  (*A)[5][12] = 0.0;

  (*A)[6][0] = 29443841.0 / 614563906.0;
  (*A)[6][1] = 0.0;
  (*A)[6][2] = 0.0;
  (*A)[6][3] = 77736583.0 / 692538347.0;
  (*A)[6][4] = -28693883.0 / 1125000000.0;
  (*A)[6][5] = 23124283.0 / 1800000000.0;
  (*A)[6][6] = 0.0;
  (*A)[6][7] = 0.0;
  (*A)[6][8] = 0.0;
  (*A)[6][9] = 0.0;
  (*A)[6][10] = 0.0;
  (*A)[6][11] = 0.0;
  (*A)[6][12] = 0.0;

  (*A)[7][0] = 16016141.0 / 946692911.0;
  (*A)[7][1] = 0.0;
  (*A)[7][2] = 0.0;
  (*A)[7][3] = 61564180.0 / 158732637.0;
  (*A)[7][4] = 22789713.0 / 633445777.0;
  (*A)[7][5] = 545815736.0 / 2771057229.0;
  (*A)[7][6] = -180193667.0 / 1043307555.0;
  (*A)[7][7] = 0.0;
  (*A)[7][8] = 0.0;
  (*A)[7][9] = 0.0;
  (*A)[7][10] = 0.0;
  (*A)[7][11] = 0.0;
  (*A)[7][12] = 0.0;

  (*A)[8][0] = 39632708.0 / 573591083.0;
  (*A)[8][1] = 0.0;
  (*A)[8][2] = 0.0;
  (*A)[8][3] = -433636366.0 / 683701615.0;
  (*A)[8][4] = -421739975.0 / 2616292301.0;
  (*A)[8][5] = 100302831.0 / 723423059.0;
  (*A)[8][6] = 790204164.0 / 839813087.0;
  (*A)[8][7] = 800635310.0 / 3783071287.0;
  (*A)[8][8] = 0.0;
  (*A)[8][9] = 0.0;
  (*A)[8][10] = 0.0;
  (*A)[8][11] = 0.0;
  (*A)[8][12] = 0.0;

  (*A)[9][0] = 246121993.0 / 1340847787.0;
  (*A)[9][1] = 0.0;
  (*A)[9][2] = 0.0;
  (*A)[9][3] = -37695042795.0 / 15268766246.0;
  (*A)[9][4] = -309121744.0 / 1061227803.0;
  (*A)[9][5] = -12992083.0 / 490766935.0;
  (*A)[9][6] = 6005943493.0 / 2108947869.0;
  (*A)[9][7] = 393006217.0 / 1396673457.0;
  (*A)[9][8] = 123872331.0 / 1001029789.0;
  (*A)[9][9] = 0.0;
  (*A)[9][10] = 0.0;
  (*A)[9][11] = 0.0;
  (*A)[9][12] = 0.0;

  (*A)[10][0] = -1028468189.0 / 846180014.0;
  (*A)[10][1] = 0.0;
  (*A)[10][2] = 0.0;
  (*A)[10][3] = 8478235783.0 / 508512852.0;
  (*A)[10][4] = 1311729495.0 / 1432422823.0;
  (*A)[10][5] = -10304129995.0 / 1701304382.0;
  (*A)[10][6] = -48777925059.0 / 3047939560.0;
  (*A)[10][7] = 15336726248.0 / 1032824649.0;
  (*A)[10][8] = -45442868181.0 / 3398467696.0;
  (*A)[10][9] = 3065993473.0 / 597172653.0;
  (*A)[10][10] = 0.0;
  (*A)[10][11] = 0.0;
  (*A)[10][12] = 0.0;

  (*A)[11][0] = 185892177.0 / 718116043.0;
  (*A)[11][1] = 0.0;
  (*A)[11][2] = 0.0;
  (*A)[11][3] = -3185094517.0 / 667107341.0;
  (*A)[11][4] = -477755414.0 / 1098053517.0;
  (*A)[11][5] = -703635378.0 / 230739211.0;
  (*A)[11][6] = 5731566787.0 / 1027545527.0;
  (*A)[11][7] = 5232866602.0 / 850066563.0;
  (*A)[11][8] = -4093664535.0 / 808688257.0;
  (*A)[11][9] = 3962137247.0 / 1805957418.0;
  (*A)[11][10] = 65686358.0 / 487910083.0;
  (*A)[11][11] = 0.0;
  (*A)[11][12] = 0.0;

  (*A)[12][0] = 403863854.0 / 491063109.0;
  (*A)[12][1] = 0.0;
  (*A)[12][2] = 0.0;
  (*A)[12][3] = -5068492393.0 / 434740067.0;
  (*A)[12][4] = -411421997.0 / 543043805.0;
  (*A)[12][5] = 652783627.0 / 914296604.0;
  (*A)[12][6] = 11173962825.0 / 925320556.0;
  (*A)[12][7] = -13158990841.0 / 6184727034.0;
  (*A)[12][8] = 3936647629.0 / 1978049680.0;
  (*A)[12][9] = -160528059.0 / 685178525.0;
  (*A)[12][10] = 248638103.0 / 1413531060.0;
  (*A)[12][11] = 0.0;
  (*A)[12][12] = 0.0;
}

/******************************************************************************/
