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

#include "threads.h"

/******************************************************************************/
/* Creation, execution and destruction of parallel threads                    */
/******************************************************************************/

void run_threads(int threads, thread_function_t * func, void **arglist)
{
  pthread_t *tid;
  int i;

  tid = MALLOC(threads, pthread_t);

  for (i = 1; i < threads; i++)
  {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    pthread_create(tid + i, &attr, func, arglist[i]);
    pthread_attr_destroy(&attr);
  }

  func(arglist[0]);

  for (i = 1; i < threads; i++)
    pthread_join(tid[i], NULL);

  FREE(tid);
}

/******************************************************************************/
/* Reduction operation                                                        */
/******************************************************************************/

double reduction_max(reduction_t * red, double value)
{
  pthread_mutex_lock(&red->lock);

  while (red->in_use)
    pthread_cond_wait(&red->used, &red->lock);

  red->thread_count++;

  if (red->thread_count == 1)
    red->value = value;
  else
    red->value = fmax(red->value, value);

  if (red->thread_count == red->threads)
  {
    red->in_use = 1;
    pthread_cond_broadcast(&red->not_all);
  }
  else
    pthread_cond_wait(&red->not_all, &red->lock);

  red->thread_count--;

  if (red->thread_count == 0)
  {
    red->in_use = 0;
    pthread_mutex_unlock(&red->lock);
    pthread_cond_broadcast(&red->used);
  }
  else
    pthread_mutex_unlock(&red->lock);

  return red->value;
}

void reduction_init(reduction_t * red, int threads)
{
  pthread_mutex_init(&red->lock, NULL);
  pthread_cond_init(&red->not_all, NULL);
  pthread_cond_init(&red->used, NULL);

  red->threads = threads;
  red->thread_count = 0;
  red->in_use = 0;
  red->value = 0.0;
}

void reduction_destroy(reduction_t * red)
{
  pthread_mutex_destroy(&red->lock);
  pthread_cond_destroy(&red->not_all);
  pthread_cond_destroy(&red->used);

  red->threads = 0;
  red->thread_count = 0;
  red->in_use = 0;
  red->value = 0.0;
}

/******************************************************************************/
/* Barrier operation                                                          */
/******************************************************************************/

#ifndef __USE_XOPEN2K

void barrier_wait(barrier_t * bar)
{
  pthread_mutex_lock(&bar->lock);

  while (bar->in_use)
    pthread_cond_wait(&bar->used, &bar->lock);

  bar->thread_count++;
  if (bar->thread_count == bar->threads)
  {
    bar->in_use = 1;
    pthread_cond_broadcast(&bar->not_all);
  }
  else
    pthread_cond_wait(&bar->not_all, &bar->lock);

  bar->thread_count--;
  if (bar->thread_count == 0)
  {
    bar->in_use = 0;
    pthread_mutex_unlock(&bar->lock);
    pthread_cond_broadcast(&bar->used);
  }
  else
    pthread_mutex_unlock(&bar->lock);
}

void barrier_init(barrier_t * bar, int threads)
{
  pthread_mutex_init(&bar->lock, NULL);
  pthread_cond_init(&bar->not_all, NULL);
  pthread_cond_init(&bar->used, NULL);

  bar->threads = threads;
  bar->thread_count = 0;
  bar->in_use = 0;
}

void barrier_destroy(barrier_t * bar)
{
  pthread_mutex_destroy(&bar->lock);
  pthread_cond_destroy(&bar->not_all);
  pthread_cond_destroy(&bar->used);

  bar->threads = 0;
  bar->thread_count = 0;
  bar->in_use = 0;
}

#endif /* __USE_XOPEN2K */

/******************************************************************************/
