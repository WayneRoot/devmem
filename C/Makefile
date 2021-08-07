ifdef ROOTDIR
PATH	:= $(PATH):$(ROOTDIR)/tools
ROMFSDIR	= $(ROOTDIR)/romfs
ROMFSINST	= romfs-inst.sh
UCLINUX_BUILD_USER = 1
include	$(ROOTDIR)/.config
LIBCDIR	= $(CONFIG_LIBCDIR)
include	$(ROOTDIR)/config.arch
#CFLAGS	= -Wall -Wextra -O2 -march=armv5te
LFLAGS	=
else
CC=gcc
CFLAGS	= -Wall -Wextra -O2
LFLAGS	=
endif

TARGET = devmem2

all: $(TARGET)

$(TARGET): devmem2.o
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

.PHONY: clean
clean:
	$(RM) *~ *.o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

romfs:
ifdef ROOTDIR
	$(ROMFSINST) /usr/bin/$(TARGET)
else
	@echo "ERROR: ROOTDIR is not set"
	@exit 1
endif
