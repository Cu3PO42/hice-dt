.PHONY: clear

CPPFLAGS += -I ../include
CPPFLAGS += -I ../../hice-dt/include
CPPFLAGS += -I ../../z3-4.7.1/src/api
CPPFLAGS += -I ../../z3-4.7.1/src/api/c++
CPPFLAGS += -L ../../z3-4.7.1/build
CPPFLAGS += -std=c++14 -Ofast

OMPFLAGS = -fopenmp

ifneq ($(OS),Windows_NT)
ifeq ($(shell uname -s),Darwin)
OMPFLAGS = -Xpreprocessor -fopenmp
endif
endif

all: chc_verifier

chc_verifier:
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(OMPFLAGS) -o chc_verifier chc_verifier.cpp ../../hice-dt/build/libhice-dt.a ../../z3-4.7.1/build/z3.a

clean:
	rm -f *.o
	rm -f *.exe
	rm -f chc_verifier
