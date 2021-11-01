Contents
========

This collection contains a set of sequential and parallel implementations of
(embedded) Runge-Kutta solvers.  These implementations are related to the ones
discussed in

1. Matthias Korch and Thomas Rauber. Optimizing locality and scalability of
   embedded Runge-Kutta solvers using block-based pipelining. _Journal of
   Parallel and Distributed Computing_, 66(3):444–468, 2006.  
   DOI: [10.1016/j.jpdc.2005.09.003](https://doi.org/10.1016/j.jpdc.2005.09.003)

2. Matthias Korch and Thomas Rauber. Parallel Low-Storage Runge-Kutta Solvers
   for ODE Systems with Limited Access Distance. _International Journal of High
   Performance Computing Applications_, 25(2):236-255, 2011.  
   DOI: [10.1177/1094342010384418](https://doi.org/10.1177/1094342010384418)

3. Matthias Korch and Thomas Rauber. _Parallel Low-Storage Runge-Kutta Solvers
   for ODE Systems with Limited Access Distance_. Bayreuth Reports on Parallel
   and Distributed Systems, No. 1, University of Bayreuth, July 2010.  
   URN: [urn:nbn:de:bvb:703-opus-7136](http://nbn-resolving.org/urn:nbn:de:bvb:703-opus-7136)

Since the publication of those articles, new implementations and performance
improvements have been added, as well as additional method tables (including
fixed step size) and other features.

The following implementations are part of this collection:


Sequential implementations
--------------------------

- `src/impl/seq/A.c`

  * suitable for general ODE systems
  * loop structure exploits spatial locality

- `src/impl/seq/D.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of reads (fused right hand side)

- `src/impl/seq/E.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes

- `src/impl/seq/F.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes (fused right hand
    side)

- `src/impl/seq/Dblock.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of reads and spatial locality

- `src/impl/seq/AEblock.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes and spatial locality

- `src/impl/seq/Fblock.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes and spatial locality
    (fused right hand side)

- `src/impl/seq/PipeD.c`

  * pipelining scheme based on implementation D
  * only suitable for ODE systems with limited access distance

- `src/impl/seq/PipeDls.c`

  * low-storage pipelining scheme based on implementation D
  * only suitable for ODE systems with limited access distance


Parallel implementations for shared address space
-------------------------------------------------

- `src/impl/pthreads/A.c`

  * suitable for general ODE systems
  * loop structure exploits spatial locality
  * shared data structures
  * barrier synchronization between the stages

- `src/impl/pthreads/D.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of reads
  * shared data structures
  * barrier synchronization between the stages

- `src/impl/pthreads/E.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes
  * shared data structures
  * barrier synchronization between the stages

- `src/impl/seq/F.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes (fused right hand
    side)
  * shared data structures
  * barrier synchronization between the stages

- `src/impl/pthreads/Dblock.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of reads and spatial locality
  * shared data structures
  * barrier synchronization between the stages

- `src/impl/pthreads/AEblock.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes and spatial locality
  * shared data structures
  * barrier synchronization between the stages

- `src/impl/seq/Fblock.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes and spatial locality
    (fused right hand side)
  * shared data structures
  * barrier synchronization between the stages

- `src/impl/pthreads/Dbc.c`

  * loop structure exploits temporal locality of reads
  * shared data structures
  * block-based synchronization between neighbors using mutex variables
  * only suitable for ODE systems with limited access distance

- `src/impl/pthreads/Dbcblock.c`

  * loop structure exploits temporal locality of reads and spatial locality
  * shared data structures
  * block-based synchronization between neighbors using mutex variables
  * only suitable for ODE systems with limited access distance

- `src/impl/pthreads/PipeD.c`

  * pipelining scheme based on implementation D
  * shared data structures
  * block-based synchronization between neighbors using mutex variables
  * only suitable for ODE systems with limited access distance

- `src/impl/pthreads/PipeD2.c`

  * pipelining scheme based on implementation D
  * alternative finalization strategy
  * shared data structures
  * only one barrier operation after the initialization of the pipelines
  * only suitable for ODE systems with limited access distance

- `src/impl/pthreads/PipeD4.c`

  * pipelining scheme based on implementation D
  * alternative computation order
  * distributed data structures
  * block-based synchronization between neighbors using mutex variables
  * only suitable for ODE systems with limited access distance

- `src/impl/pthreads/PipeD4ls.c`

  * low-storage pipelining scheme based on implementation D
  * alternative computation order
  * distributed data structures
  * block-based synchronization between neighbors using mutex variables
  * only suitable for ODE systems with limited access distance

These implementations require a POSIX thread library to be installed.


Parallel implementations for distributed address space 
------------------------------------------------------

- `src/impl/mpi/A.c`

  * suitable for general ODE systems
  * loop structure exploits spatial locality
  * multibroadcast operations (`MPI_Allgatherv`) between the stages

- `src/impl/mpi/D.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of reads
  * multibroadcast operations (`MPI_Allgatherv`) between the stages

- `src/impl/mpi/E.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes
  * multibroadcast operations (`MPI_Allgatherv`) between the stages

- `src/impl/seq/F.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes (fused right hand
    side)
  * multibroadcast operations (`MPI_Allgatherv`) between the stages

- `src/impl/mpi/Dblock.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of reads and spatial locality
  * multibroadcast operations (`MPI_Allgatherv`) between the stages

- `src/impl/mpi/AEblock.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes and spatial locality
  * multibroadcast operations (`MPI_Allgatherv`) between the stages

- `src/impl/seq/Fblock.c`

  * suitable for general ODE systems
  * loop structure exploits temporal locality of writes and spatial locality
    (fused right hand side)
  * multibroadcast operations (`MPI_Allgatherv`) between the stages

- `src/impl/mpi/Dbc.c`

  * loop structure exploits temporal locality of reads
  * block-based communication between neighbors using single transfer
    operations (`MPI_Isend`/`MPI_Irecv`)
  * only suitable for ODE systems with limited access distance

- `src/impl/mpi/Dbcblock.c`

  * loop structure exploits temporal locality of reads and spatial locality
  * block-based communication between neighbors using single transfer
    operations (`MPI_Isend`/`MPI_Irecv`)
  * only suitable for ODE systems with limited access distance

- `src/impl/mpi/PipeD.c`

  * pipelining scheme based on implementation D
  * block-based communication between neighbors using single transfer
    operations (`MPI_Isend`/`MPI_Irecv`)
  * only suitable for ODE systems with limited access distance

- `src/impl/mpi/PipeD2.c`

  * pipelining scheme based on implementation D
  * alternative finalization strategy
  * only one pair of single transfer operations (`MPI_Isend`/`MPI_Irecv`), but
    more data is transferred
  * only suitable for ODE systems with limited access distance

- `src/impl/mpi/PipeD4.c`

  * pipelining scheme based on implementation D
  * alternative computation order
  * block-based communication between neighbors using single transfer
    operations (`MPI_Isend`/`MPI_Irecv`)
  * only suitable for ODE systems with limited access distance

- `src/impl/mpi/PipeD5.c`

  * pipelining scheme based on implementation D
  * alternative computation order
  * block-based communication between neighbors using single transfer
    operations (`MPI_Isend`/`MPI_Irecv`)
  * only suitable for ODE systems with limited access distance

- `src/impl/mpi/PipeD4ls.c`

  * low-storage pipelining scheme based on implementation D
  * alternative computation order
  * block-based communication between neighbors using single
    transfer operations (`MPI_Isend`/`MPI_Irecv`)
  * only suitable for ODE systems with limited access distance

These implementations require an MPI library to be installed.


Test problems
-------------

Only one test problem is provided, but you can add your own test
problems easily.

- BRUSS2D-MIX
  * 2D Brusselator equation with diffusion (2D PDE with 2 variables)
  * semi-discretized on spatial N x N grid
  * mixed row-oriented ordering to obtain a limited access distance of d(f)=2N


Configuration of parameters
===========================

Some parameters can be changed by editing the file `include/config.h`. See the
comments in that file for details. 

You can also overwrite the settings in `include/config.h` by using the make
variable `DEFINES`, e.g.,

    > make DEFINES="PROBLEM=LV METHOD=VERNER65"

But don't forget to `make clean` first.

Compiling
=========

A Makefile is provided which builds all combinations of test problems and
implementations automatically when you invoke `make` &ndash; provided you have gcc
and MPI installed. The binaries will be created in the `bin/` directory.

If MPI is not installed on your system, you should set `HAVE_MPI` to `no` in
the `Makefile`.

When you modify the source codes, in particular, when you add or remove
implementations or when you add or remove `#include` directives, it will be
necessary to run `make dep` to let `make` know the new dependencies between
the source files.


Testing and running
===================

After successful compilation, you can run the sequential implementations
directly from the `bin/` directory, e.g., you can type:

    > ./bin/seq_bruss2d-mix_D

If you have built with `HAVE_MPI=yes`, some MPI libraries may require the use
of `mpirun` or `mpiexec`, e.g.:

    > mpirun -n 1 ./bin/seq_bruss2d-mix_D

The shared-address space implementations can be started in a similar fashion,
but you should specify the number of threads beforehand by setting the
environment variable `NUM_THREADS`:

    > NUM_THREADS=4 ./bin/pthreads_bruss2d-mix_Dbc

The distributed-address space implementations usually require the use of
`mpirun` or `mpiexec`:

    > mpirun -n 10 ./bin/mpi_bruss2d-mix_PipeD4ls

There are two `make` targets to help you test and evaluate the
implementations.

    > make check
	
will run all implementations and print a footprint of the final output
approximation of each implementation per screen row. The output rows are
aligned so that one can visually recognize differences in the results easily.

    > make time

will run all implementations and print the kernel time of each per screen row.


Coding style
============

All C source codes should be formatted with GNU indent using the options in
the included `.indent.pro` file before commit. Similarly, Perl source codes
should be formatted with perltidy using the options in the included
`perltidy.conf` file before commit. You can run `make indent` to format all
source codes.


License
=======

GPL version 3 or later. Please see `COPYING` for details.


Contact
=======

Prof. Dr. Matthias Korch  
Department of Computer Science  
University of Bayreuth  
95440 Bayreuth  
Germany

E-Mail: korch@uni-bayreuth.de
