TESTS  = $(shell ls {float,double}.{ispc,cpp}.test)

benchmark: all
	@ for T in $(TESTS); do echo $$T ; ./$$T; done | tee /tmp/upwindresults.txt
	 echo -e "\e[92m     "
	@ cat /tmp/upwindresults.txt | egrep PERF | awk '{print $$3,$$5}' | tr '\n' ' ' | \
	  awk '{printf("DP ISPC GAIN: %02.1fX - %02.1fX\nSP ISPC GAIN: %02.1fX - %02.1fX", $$3 / $$2, $$4 / $$1, $$7 / $$6, $$8 / $$5);}'
	@ echo -e "\e[0m"

all: 
	make -f test.Makefile all real=float
	make -f test.Makefile all real=double

clean:
	make -f test.Makefile clean

.PHONY = all clean	
