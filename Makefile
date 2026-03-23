CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
INCLUDES := -I./include -I/opt/homebrew/include/eigen3 -I./vendor -I/opt/homebrew/lib/python3.14/site-packages/pybind11/include -I/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/include/python3.14

SRC_DIR      := src
BENCH_DIR    := benchmarks
TEST_DIR     := tests
PRELIM_DIR   := tests/prelim-tests
EXAMPLES_DIR := examples

BUILD_DIR      := build
BIN_DIR        := bin

BUILD_SRC      := $(BUILD_DIR)/src
BUILD_BENCH    := $(BUILD_DIR)/benchmarks
BUILD_TEST     := $(BUILD_DIR)/tests
BUILD_PRELIM   := $(BUILD_DIR)/tests/prelim-tests
BUILD_EXAMPLES := $(BUILD_DIR)/examples
PYTHON_EXT := optim_engine$(shell python3-config --extension-suffix)

SRC_SRCS     := $(shell find $(SRC_DIR) -name "*.cpp" 2>/dev/null)
BENCH_SRCS   := $(wildcard $(BENCH_DIR)/*.cpp)
TEST_SRCS    := $(wildcard $(TEST_DIR)/*.cpp)
PRELIM_SRCS  := $(wildcard $(PRELIM_DIR)/*.cpp)
EXAMPLE_SRCS := $(wildcard $(EXAMPLES_DIR)/*.cpp)

SRC_OBJS     := $(patsubst $(SRC_DIR)/%.cpp,     $(BUILD_SRC)/%.o,      $(SRC_SRCS))
BENCH_OBJS   := $(patsubst $(BENCH_DIR)/%.cpp,   $(BUILD_BENCH)/%.o,    $(BENCH_SRCS))
TEST_OBJS    := $(patsubst $(TEST_DIR)/%.cpp,    $(BUILD_TEST)/%.o,     $(TEST_SRCS))
PRELIM_OBJS  := $(patsubst $(PRELIM_DIR)/%.cpp,  $(BUILD_PRELIM)/%.o,   $(PRELIM_SRCS))
EXAMPLE_OBJS := $(patsubst $(EXAMPLES_DIR)/%.cpp,$(BUILD_EXAMPLES)/%.o, $(EXAMPLE_SRCS))


BENCH_BIN    := $(BIN_DIR)/benchmarks
TEST_BIN     := $(BIN_DIR)/tests
PRELIM_BIN   := $(BIN_DIR)/prelim-tests

EXAMPLE_BINS := $(patsubst $(EXAMPLES_DIR)/%.cpp, $(BIN_DIR)/%, $(EXAMPLE_SRCS))

.PHONY: all src benchmarks tests prelim-tests examples clean help

all: src tests prelim-tests examples

src: $(SRC_OBJS)

benchmarks: $(BENCH_BIN)

$(BENCH_BIN): $(SRC_OBJS) $(BENCH_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@


tests: $(TEST_BIN)

$(TEST_BIN): $(SRC_OBJS) $(TEST_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

prelim-tests: $(PRELIM_BIN)

$(PRELIM_BIN): $(SRC_OBJS) $(PRELIM_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@


examples: $(EXAMPLE_BINS)

$(BIN_DIR)/%: $(BUILD_EXAMPLES)/%.o $(SRC_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_SRC)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_BENCH)/%.o: $(BENCH_DIR)/%.cpp | $(BUILD_BENCH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_TEST)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_TEST)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_PRELIM)/%.o: $(PRELIM_DIR)/%.cpp | $(BUILD_PRELIM)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_EXAMPLES)/%.o: $(EXAMPLES_DIR)/%.cpp | $(BUILD_EXAMPLES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


$(BIN_DIR) $(BUILD_BENCH) $(BUILD_TEST) $(BUILD_PRELIM) $(BUILD_EXAMPLES):
	@mkdir -p $@


python: $(PYTHON_EXT)

$(PYTHON_EXT): python/bindings.cpp $(SRC_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		-shared -fPIC -undefined dynamic_lookup $^ -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

help:
	@echo ""
	@echo "  all          Build everything"
	@echo "  src          Compile src/ objects only (no linking)"
	@echo "  benchmarks   Build bin/benchmarks"
	@echo "  tests        Build bin/tests"
	@echo "  prelim-tests Build bin/prelim-tests"
	@echo "  examples     Build bin/<name> for each examples/*.cpp"
	@echo "  clean        Remove build/ and bin/"
	@echo ""
