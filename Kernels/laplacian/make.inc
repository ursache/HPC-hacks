CC=icc
CFLAGS=-O3 -g -march=native -vec-report6 -std=gnu99

#CFLAGS += -fopenmp
#CFLAGS += -DSTATIC
#CFLAGS += -DMPI
#CFLAGS += -DPAPI -lpapi
CFLAGS += -opt-streaming-stores always
CFLAGS += -L/users/bcumming/repos/lib/lib 
CFLAGS += -I/users/bcumming/repos/lib/include
CFLAGS += -I/users/fgilles/bin/iaca-lin64/include/
#CFLAGS += -DIACA
CFLAGS += -DSTREAMSTORE

