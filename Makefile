################################################################################
# This file is part of a collection of embedded Runge-Kutta solvers.           #
# Copyright (C) 2009-2030, Matthias Korch, University of Bayreuth, Germany.    #
#                                                                              #
# This program is free software: you can redistribute it and/or modify         #
# it under the terms of the GNU General Public License as published by         #
# the Free Software Foundation, either version 3 of the License, or            #
# (at your option) any later version.                                          #
#                                                                              #
# This program is distributed in the hope that it will be useful,              #
# but WITHOUT ANY WARRANTY; without even the implied warranty of               #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                #
# GNU General Public License for more details.                                 #
#                                                                              #
# You should have received a copy of the GNU General Public License            #
# along with this program.  If not, see <https://www.gnu.org/licenses/>.       #
################################################################################

HAVE_MPI        = yes

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

DEPFILE         = .depend

CORES           = 4

################################################################################

CFLAGS  = -I$(INCDIR) -D_REENTRANT -Wall -O3 -g -D_GNU_SOURCE
ifeq "$(HAVE_MPI)" "yes"
CC      = mpicc
CFLAGS += -DHAVE_MPI
else
CC      = gcc
endif
LDFLAGS = -lpthread -lm
MAKEDEP = $(CC) $(CFLAGS) -MM -MG

################################################################################

HEADERS          = $(wildcard $(INCDIR)/*.h)
PROB_SOURCES     = $(wildcard $(PROBDIR)/*.c)
IMPL_SOURCES_SEQ = $(wildcard $(SEQDIR)/*.c)
IMPL_SOURCES_PTH = $(wildcard $(PTHDIR)/*.c)
ifeq "$(HAVE_MPI)" "yes"
IMPL_SOURCES_MPI = $(wildcard $(MPIDIR)/*.c)
else
IMPL_SOURCES_MPI =
endif
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

$(ALL_TARGETS): $(BINDIR)
		$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

################################################################################

%.o:		%.c
		$(CC) $(CFLAGS) -o $@ -c $<

################################################################################

$(BINDIR):
		mkdir $@

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
done > $(DEPFILE)
		for i in $(IMPL_SEQ); do \
  for p in $(PROBLEMS); do \
    echo "bin/$(PREFIX_SEQ)_$${p}_$${i}: $(OTHER_OBJECTS) \\" ; \
    echo "  $(SEQDIR)/$$i.o $(PROBDIR)/$$p.o" ; \
    echo ; \
  done ; \
done >> $(DEPFILE)
		for i in $(IMPL_PTH); do \
  for p in $(PROBLEMS); do \
    echo "bin/$(PREFIX_PTH)_$${p}_$${i}: $(OTHER_OBJECTS) \\" ; \
    echo "  $(PTHDIR)/$$i.o $(PROBDIR)/$$p.o" ; \
    echo ; \
  done ; \
done >> $(DEPFILE)
		for i in $(IMPL_MPI); do \
  for p in $(PROBLEMS); do \
    echo "bin/$(PREFIX_MPI)_$${p}_$${i}: $(OTHER_OBJECTS) \\" ; \
    echo "  $(MPIDIR)/$$i.o $(PROBDIR)/$$p.o" ; \
    echo ; \
  done ; \
done >> $(DEPFILE)

################################################################################

.PHONY:	check
check:	all
	for P in $(SEQ_TARGETS); do \
  printf "%35s\t" ./$$P ; \
  ./$$P | grep "Result" ; \
done ; \
for P in $(PTH_TARGETS); do \
  printf "%35s\t" ./$$P ; \
  NUM_THREADS=$(CORES) ./$$P | grep "Result" ; \
done ; \
for P in $(MPI_TARGETS); do \
  printf "%35s\t" ./$$P ; \
  mpirun -n $(CORES) ./$$P | grep "Result" ; \
done

################################################################################

.PHONY:	time
time:	all
	for P in $(SEQ_TARGETS); do \
  printf "%35s\t" ./$$P ; \
  ./$$P | grep "Kernel time" ; \
done ; \
echo ; \
for P in $(PTH_TARGETS); do \
  printf "%35s\t" ./$$P ; \
  NUM_THREADS=$(CORES) ./$$P | grep "Kernel time" ; \
done ; \
echo ; \
for P in $(MPI_TARGETS); do \
  printf "%35s\t" ./$$P ; \
  mpirun -n $(CORES) ./$$P | grep "Kernel time" ; \
done

################################################################################

.PHONY:	indent
indent:
	indent $(HEADERS) $(ALL_SOURCES)

################################################################################

-include $(DEPFILE)

################################################################################
