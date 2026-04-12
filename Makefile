unexport SDKROOT
CXX      := clang++
CXXFLAGS := -std=c++20 -Wall -Wextra -O2
INCLUDES := -I./include -I/opt/homebrew/include/eigen3 -I./vendor -I/opt/homebrew/lib/python3.14/site-packages/pybind11/include -I/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/include/python3.14

SRC_DIR      := src
BENCH_DIR    := benchmarks
PRELIM_DIR   := tests/prelim-tests
EXAMPLES_DIR := examples

BUILD_DIR      := build
BIN_DIR        := bin

BUILD_SRC      := $(BUILD_DIR)/src
BUILD_BENCH    := $(BUILD_DIR)/benchmarks
BUILD_PRELIM   := $(BUILD_DIR)/tests/prelim-tests
BUILD_EXAMPLES := $(BUILD_DIR)/examples

PYTHON_EXT := optim_engine$(shell python3-config --extension-suffix)

SRC_SRCS     := $(shell find $(SRC_DIR) -name "*.cpp" 2>/dev/null)
BENCH_SRCS   := $(wildcard $(BENCH_DIR)/*.cpp)
PRELIM_SRCS  := $(wildcard $(PRELIM_DIR)/*.cpp)
EXAMPLE_SRCS := $(wildcard $(EXAMPLES_DIR)/*.cpp)

SRC_OBJS     := $(patsubst $(SRC_DIR)/%.cpp,      $(BUILD_SRC)/%.o,      $(SRC_SRCS))
BENCH_OBJS   := $(patsubst $(BENCH_DIR)/%.cpp,    $(BUILD_BENCH)/%.o,    $(BENCH_SRCS))
PRELIM_OBJS  := $(patsubst $(PRELIM_DIR)/%.cpp,   $(BUILD_PRELIM)/%.o,   $(PRELIM_SRCS))
EXAMPLE_OBJS := $(patsubst $(EXAMPLES_DIR)/%.cpp, $(BUILD_EXAMPLES)/%.o, $(EXAMPLE_SRCS))

BENCH_BIN    := $(BIN_DIR)/benchmarks
PRELIM_BINS  := $(patsubst $(PRELIM_DIR)/%.cpp, $(BIN_DIR)/prelim-tests/%, $(PRELIM_SRCS))
EXAMPLE_BINS := $(patsubst $(EXAMPLES_DIR)/%.cpp, $(BIN_DIR)/%, $(EXAMPLE_SRCS))

.PRECIOUS: $(BUILD_PRELIM)/%.o $(BUILD_BENCH)/%.o $(BUILD_EXAMPLES)/%.o

.PHONY: all src benchmarks prelim-tests run-tests examples python clean help

all: src benchmarks prelim-tests examples python

src: $(SRC_OBJS)

benchmarks: $(BENCH_BIN)

$(BENCH_BIN): $(SRC_OBJS) $(BENCH_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

prelim-tests: $(PRELIM_BINS)

$(BIN_DIR)/prelim-tests/%: $(BUILD_PRELIM)/%.o $(SRC_OBJS) | $(BIN_DIR)/prelim-tests
	$(CXX) $(CXXFLAGS) $^ -o $@

run-tests: prelim-tests
	@for t in $(PRELIM_BINS); do echo "\n--- $$t ---"; $$t; done

examples: $(EXAMPLE_BINS)

$(BIN_DIR)/%: $(BUILD_EXAMPLES)/%.o $(SRC_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_SRC)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_BENCH)/%.o: $(BENCH_DIR)/%.cpp | $(BUILD_BENCH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_PRELIM)/%.o: $(PRELIM_DIR)/%.cpp | $(BUILD_PRELIM)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_EXAMPLES)/%.o: $(EXAMPLES_DIR)/%.cpp | $(BUILD_EXAMPLES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR) $(BIN_DIR)/prelim-tests $(BUILD_BENCH) $(BUILD_PRELIM) $(BUILD_EXAMPLES):
	@mkdir -p $@

python: $(PYTHON_EXT)

$(PYTHON_EXT): python/bindings.cpp $(SRC_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		-shared -fPIC -undefined dynamic_lookup $^ -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

help:
	@echo ""
	@echo "  all          Build everything (src, benchmarks, prelim-tests, examples, python)"
	@echo "  src          Compile src/ objects only"
	@echo "  benchmarks   Build bin/benchmarks"
	@echo "  prelim-tests Build bin/prelim-tests/<name> for each test"
	@echo "  run-tests    Build and run all prelim tests"
	@echo "  examples     Build bin/<name> for each examples/*.cpp"
	@echo "  python       Build Python extension"
	@echo "  clean        Remove build/ and bin/"
	@echo ""
