# Build both drivers with mpicc. The serial driver makes no MPI calls, so it
# can also be built with a plain compiler: `make SERIAL_CC=cc`.
MPICC     ?= mpicc
SERIAL_CC ?= $(MPICC)

# On macOS, Homebrew's libmpi is built for the running OS version; match it so
# the linker does not warn about a deployment-target mismatch.
ifeq ($(shell uname -s),Darwin)
  export MACOSX_DEPLOYMENT_TARGET ?= $(shell sw_vers -productVersion)
endif

BUILD ?= release
ifeq ($(BUILD),debug)
  OPTFLAGS = -O0 -g -fsanitize=address -fno-omit-frame-pointer
  LDFLAGS += -fsanitize=address
  BINDIR   = bin/debug
else
  OPTFLAGS = -O3
  BINDIR   = bin
endif

# -ffp-contract=off: never fuse a*b+c into an FMA. Clang fuses by default and
# GCC does not, so this keeps float results the same across compilers.
CFLAGS  = $(OPTFLAGS) -Wall -Wextra -std=c11 -ffp-contract=off -Iinclude
LDLIBS  = -lm
OBJDIR  = build/$(BUILD)

COMMON  = bmp.o filters.o cli.o
SERIAL_OBJS = $(addprefix $(OBJDIR)/,$(COMMON) timing.o main_serial.o)
MPI_OBJS    = $(addprefix $(OBJDIR)/,$(COMMON) timing_mpi.o decomp.o main_mpi.o)

.PHONY: all serial mpi debug test test-debug clean

all: serial mpi

serial: $(BINDIR)/filter_serial $(BINDIR)/bmp_roundtrip

mpi: $(BINDIR)/filter_mpi

$(BINDIR)/filter_mpi: $(MPI_OBJS) | $(BINDIR)
	$(MPICC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Files that include <mpi.h> must be compiled by the MPI wrapper.
$(OBJDIR)/decomp.o $(OBJDIR)/main_mpi.o: $(OBJDIR)/%.o: src/%.c $(wildcard include/*.h) | $(OBJDIR)
	$(MPICC) $(CFLAGS) -c -o $@ $<

# timing.c is built twice: MPI_Wtime for the MPI driver, clock_gettime for serial.
$(OBJDIR)/timing_mpi.o: src/timing.c include/timing.h | $(OBJDIR)
	$(MPICC) $(CFLAGS) -DUSE_MPI -c -o $@ $<

debug:
	$(MAKE) BUILD=debug all

$(BINDIR)/filter_serial: $(SERIAL_OBJS) | $(BINDIR)
	$(SERIAL_CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BINDIR)/bmp_roundtrip: $(OBJDIR)/bmp.o $(OBJDIR)/bmp_roundtrip.o | $(BINDIR)
	$(SERIAL_CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o: src/%.c $(wildcard include/*.h) | $(OBJDIR)
	$(SERIAL_CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/bmp_roundtrip.o: tests/bmp_roundtrip.c include/bmp.h | $(OBJDIR)
	$(SERIAL_CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR) $(BINDIR):
	mkdir -p $@

test: all
	scripts/run_tests.sh

# Same suite against the AddressSanitizer build.
test-debug: debug
	BIN_DIR=bin/debug scripts/run_tests.sh

clean:
	rm -rf build bin
