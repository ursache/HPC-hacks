arch=intel

EXEC=zdotu

zdotu_SRCS= zdotu.cpp 

##zdotu_ ne fonctionne pas avec les flags suivant :
ifeq ($(arch),gnu)
CXX = g++ -std=c++11 
ERRORS = -Wall -Wextra -pedantic
OPTION = -L${MKL_ROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread 
endif

##zdotu_ fonctionne avec les flags suivant :
ifeq ($(arch),intel)
CXX = icpc -std=c++11 
ERRORS = -g -O3 -Wall -Wextra -pedantic
OPTION = -mkl
endif

#-----------------------------------------------------------------

CXXFLAGS = $(ERRORS) $(OPTION)
LDFLAGS  = $(ERRORS) $(OPTION)

BUILD=./build

SRCS=$(wildcard *.cpp)

all: OPTION += -O3 -DNDEBUG -I/users/fgilles/bin/iaca-lin64/include/
all:$(EXEC)
        $(POSTPROCESS)

debug: OPTION += -ggdb
debug:$(EXEC)
        $(POSTPROCESS)

.SECONDEXPANSION:
$(EXEC): $$(patsubst %.cpp, $(BUILD)/%.o, $$($$@_SRCS))
	@echo Links $(notdir $^)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(BUILD)/%.o:%.cpp
	@mkdir -p $(BUILD)
	@echo Creates $(notdir $@)
	$(CXX) -MD -c $(CXXFLAGS) $< -o $@

-include $(addprefix $(BUILD)/,$(SRCS:.cpp=.d))

clean:
	rm -rf $(BUILD) $(EXEC)

ref:
	@echo Create the documentation
	@doxygen doxygen/Doxyfile > doxygen/log
	@firefox doxygen/html/files.html &

