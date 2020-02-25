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

#ifndef THREADS_H_
#define THREADS_H_

/******************************************************************************/

#include "config.h"
#include "tools.h"

#include <pthread.h>

/******************************************************************************/
/* Mutex variables                                                            */
/******************************************************************************/

#define MUTEX     0
#define SPINLOCK  1

#if (LOCKTYPE == SPINLOCK) && defined(__USE_XOPEN2K)

typedef struct
{
  PADDING_FIELD(pad1);
  pthread_spinlock_t lock;
    PADDING_FIELD(pad2);
} mutex_lock_t;

#if PAD_SIZE > 0
#define MUTEX_LOCK_INITIALIZER    { "", __LT_SPINLOCK_INIT, "" }
#else
#define MUTEX_LOCK_INITIALIZER    { __LT_SPINLOCK_INIT }
#endif

#define mutex_lock_init(MUTEX)    pthread_spin_init(&((MUTEX)->lock), 0)
#define mutex_lock_destroy(MUTEX) pthread_spin_destroy(&((MUTEX)->lock))

#define mutex_lock_lock(MUTEX)    pthread_spin_lock(&((MUTEX)->lock))
#define mutex_lock_unlock(MUTEX)  pthread_spin_unlock(&((MUTEX)->lock))

#define mutex_lock_trylock(MUTEX) pthread_spinlock_trylock(&((MUTEX)->lock))

#else

typedef struct
{
  PADDING_FIELD(pad1);
  pthread_mutex_t lock;
    PADDING_FIELD(pad2);
} mutex_lock_t;

#if PAD_SIZE > 0
#define MUTEX_LOCK_INITIALIZER    { "", PTHREAD_MUTEX_INITIALIZER, "" }
#else
#define MUTEX_LOCK_INITIALIZER    { PTHREAD_MUTEX_INITIALIZER }
#endif

#define mutex_lock_init(MUTEX)    pthread_mutex_init(&((MUTEX)->lock), NULL)
#define mutex_lock_destroy(MUTEX) pthread_mutex_destroy(&((MUTEX)->lock))

#define mutex_lock_lock(MUTEX)    pthread_mutex_lock(&((MUTEX)->lock))
#define mutex_lock_unlock(MUTEX)  pthread_mutex_unlock(&((MUTEX)->lock))

#define mutex_lock_trylock(MUTEX) pthread_mutex_trylock(&((MUTEX)->lock))

#endif

/******************************************************************************/
/* Thread function type                                                       */
/******************************************************************************/

typedef void *(thread_function_t) (void *);

/******************************************************************************/
/* Creation, execution and destruction of parallel threads                    */
/******************************************************************************/

void run_threads(int threads, thread_function_t * func, void **arglist);

/******************************************************************************/
/* Reduction operation                                                        */
/******************************************************************************/

typedef struct
{
  PADDING_FIELD(pad1);
  int threads;
  int thread_count;
  int in_use;
  double value;
  pthread_mutex_t lock;
  pthread_cond_t not_all;
  pthread_cond_t used;
    PADDING_FIELD(pad2);
} reduction_t;

double reduction_max(reduction_t * red, double value);

void reduction_init(reduction_t * red, int n);
void reduction_destroy(reduction_t * red);

/******************************************************************************/
/* Barrier operation                                                          */
/******************************************************************************/

#ifdef __USE_XOPEN2K

typedef struct
{
  PADDING_FIELD(pad1);
  pthread_barrier_t bar;
    PADDING_FIELD(pad2);
} barrier_t;

#define barrier_wait(BARRIER)    \
  pthread_barrier_wait(&((BARRIER)->bar))

#define barrier_init(BARRIER, N) \
  pthread_barrier_init(&((BARRIER)->bar), NULL, (N))

#define barrier_destroy(BARRIER) \
  pthread_barrier_destroy(&((BARRIER)->bar))

#else /* __USE_XOPEN2K */

typedef struct
{
  PADDING_FIELD(pad1);
  int threads;
  int thread_count;
  int in_use;
  pthread_mutex_t lock;
  pthread_cond_t not_all;
  pthread_cond_t used;
    PADDING_FIELD(pad2);
} barrier_t;

void barrier_wait(barrier_t * bar);
void barrier_init(barrier_t * bar, int threads);
void barrier_destroy(barrier_t * bar);

#endif /* __USE_XOPEN2K */

/******************************************************************************/

#endif /* THREADS_H_ */

/******************************************************************************/
