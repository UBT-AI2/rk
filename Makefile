################################################################################
# This file is part of a collection of embedded Runge-Kutta solvers.           #
# Copyright (C) 2009-2010, Matthias Korch, University of Bayreuth, Germany.    #
#                                                                              #
# This is free software; you can redistribute it and/or modify it under the    #
# terms of the GNU General Public License as published by the Free Software    #
# Foundation; either version 2 of the License, or (at your option) any later   #
# version.                                                                     #
#                                                                              #
# This software is distributed in the hope that it will be useful, but         #
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY   #
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License      #
# for more details.                                                            #
#                                                                              #
# You should have received a copy of the GNU General Public License along      #
# with this program; if not, write to the Free Software Foundation, Inc.,      #
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.                #
################################################################################

BINDIR 		= bin
INCDIR 		= include
SRCDIR 		= src

PREFIX_SEQ 	= seq
PREFIX_PTH 	= pthreads
PREFIX_MPI 	= mpi

PROBDIR    	= $(SRCDIR)/problems
SEQDIR  	= $(SRCDIR)/impl/$(PREFIX_SEQ)
PTHDIR  	= $(SRCDIR)/impl/$(PREFIX_PTH)
MPIDIR  	= $(SRCDIR)/impl/$(PREFIX_MPI)

################################################################################

CC      = gcc
MPIC    = mpicc
CFLAGS  = -I$(INCDIR) -D_REENTRANT -Wall -O3 -g
LDFLAGS = -lpthread -lm
MAKEDEP = $(CC) $(CFLAGS) -MM -MG

################################################################################

PROB_SOURCES     = $(wildcard $(PROBDIR)/*.c)
IMPL_SOURCES_SEQ = $(wildcard $(SEQDIR)/*.c)
IMPL_SOURCES_PTH = $(wildcard $(PTHDIR)/*.c)
IMPL_SOURCES_MPI = $(wildcard $(MPIDIR)/*.c)
OTHER_SOURCES    = $(wildcard $(SRCDIR)/*.c)
IMPL_SOURCES     = $(IMPL_SOURCES_SEQ) $(IMPL_SOURCES_PTH) $(IMPL_SOURCES_MPI)
ALL_SOURCES      = $(IMPL_SOURCES) $(PROB_SOURCES) $(OTHER_SOURCES)

IMPL_OBJECTS     = $(IMPL_SOURCES:.c=.o)
PROB_OBJECTS     = $(PROB_SOURCES:.c=.o)
OTHER_OBJECTS    = $(OTHER_SOURCES:.c=.o)
ALL_OBJECTS      = $(IMPL_OBJECTS) $(PROB_OBJECTS) $(OTHER_OBJECTS)

PROBLEMS         = $(patsubst $(PROBDIR)/%.c, %, $(PROB_SOURCES))
IMPL_SEQ         = $(patsubst $(SEQDIR)/%.c, %, $(IMPL_SOURCES_SEQ))
IMPL_PTH         = $(patsubst $(PTHDIR)/%.c, %, $(IMPL_SOURCES_PTH))
IMPL_MPI         = $(patsubst $(MPIDIR)/%.c, %, $(IMPL_SOURCES_MPI))

SEQ_TARGETS      = $(foreach i, $(IMPL_SEQ), \
			$(foreach p, $(PROBLEMS), $(BINDIR)/$(PREFIX_SEQ)_$(p)_$(i)))
PTH_TARGETS      = $(foreach i, $(IMPL_PTH), \
			$(foreach p, $(PROBLEMS), $(BINDIR)/$(PREFIX_PTH)_$(p)_$(i)))
MPI_TARGETS      = $(foreach i, $(IMPL_MPI), \
			$(foreach p, $(PROBLEMS), $(BINDIR)/$(PREFIX_MPI)_$(p)_$(i)))

ALL_TARGETS      = $(SEQ_TARGETS) $(PTH_TARGETS) $(MPI_TARGETS)

################################################################################

.PHONY:		all
all:		$(ALL_TARGETS)

.PHONY:		$(PREFIX_SEQ)
$(PREFIX_SEQ):	$(SEQ_TARGETS)

.PHONY:		$(PREFIX_PTH)
$(PREFIX_PTH):	$(PTH_TARGETS)

.PHONY:		$(PREFIX_MPI)
$(PREFIX_MPI):	$(MPI_TARGETS)

################################################################################

$(SEQ_TARGETS) $(PTH_TARGETS):
		$(CC) -o $@ $^ $(LDFLAGS)

$(MPI_TARGETS):
		$(MPICC) -o $@ $^ $(LDFLAGS)

################################################################################

%.o:		%.c
		$(CC) $(CFLAGS) -o $@ -c $<

################################################################################

.PHONY:		clean
clean:
		-rm -f $(ALL_OBJECTS) $(ALL_TARGETS)

################################################################################

.PHONY:		dep
dep:
		for i in $(ALL_SOURCES); do \
  echo -n "$$(dirname $$i)/" ; \
  $(MAKEDEP) $$i; \
  echo ; \
done > .depend
		for i in $(IMPL_SEQ); do \
  for p in $(PROBLEMS); do \
    echo "bin/$(PREFIX_SEQ)_$${p}_$${i}: $(OTHER_OBJECTS) \\" ; \
    echo "  $(SEQDIR)/$$i.o $(PROBDIR)/$$p.o" ; \
    echo ; \
  done ; \
done >> .depend
		for i in $(IMPL_PTH); do \
  for p in $(PROBLEMS); do \
    echo "bin/$(PREFIX_PTH)_$${p}_$${i}: $(OTHER_OBJECTS) \\" ; \
    echo "  $(PTHDIR)/$$i.o $(PROBDIR)/$$p.o" ; \
    echo ; \
  done ; \
done >> .depend
		for i in $(IMPL_MPI); do \
  for p in $(PROBLEMS); do \
    echo "bin/$(PREFIX_MPI)_$${p}_$${i}: $(OTHER_OBJECTS) \\" ; \
    echo "  $(MPIDIR)/$$i.o $(PROBDIR)/$$p.o" ; \
    echo ; \
  done ; \
done >> .depend

################################################################################

include .depend

################################################################################
