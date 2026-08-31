unexport SDKROOT

# Respect CXX from the environment or the command line; otherwise keep clang++.
ifeq ($(origin CXX),default)
    CXX := clang++
endif

PYTHON ?= python3

# Dependency include paths are discovered rather than hardcoded. The previous
# absolute /opt/homebrew paths (with a pinned python3.14) existed only on one
# machine, so the build could not run anywhere else. Override if discovery
# fails, e.g. make EIGEN_INCLUDES=-I/usr/include/eigen3
EIGEN_INCLUDES ?= $(shell pkg-config --cflags eigen3 2>/dev/null)
ifeq ($(strip $(EIGEN_INCLUDES)),)
    EIGEN_INCLUDES := $(shell for d in /opt/homebrew/include/eigen3 /usr/local/include/eigen3 /usr/include/eigen3; do [ -d "$$d" ] && echo "-I$$d" && break; done)
endif

# Emits both the Python and the pybind11 include paths.
PYBIND_INCLUDES ?= $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null)

# -fPIC is required because these objects are also linked into the Python
# extension module. macOS emits position-independent code regardless, but on
# Linux linking non-PIC objects into a shared library is a hard error.
CXXFLAGS := -std=c++20 -Wall -Wextra -O2 -fPIC
# Third-party headers (Eigen, pybind11, the vendored Catch2) are included with
# -isystem so their own warnings are not attributed to this project. Eigen's
# SparseCore trips -Wunused-but-set-variable under clang, which would otherwise
# make -Werror impossible.
EIGEN_INCLUDES  := $(patsubst -I%,-isystem %,$(EIGEN_INCLUDES))
PYBIND_INCLUDES := $(patsubst -I%,-isystem %,$(PYBIND_INCLUDES))

INCLUDES := -I./include -isystem ./vendor $(EIGEN_INCLUDES) $(PYBIND_INCLUDES)

# -undefined dynamic_lookup lets the extension resolve Python symbols at load
# time on macOS. It is not a valid GNU ld flag, so Linux gets a plain -shared.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    SHARED_LDFLAGS := -shared -undefined dynamic_lookup
else
    SHARED_LDFLAGS := -shared
endif

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

# Derived through sysconfig rather than a `<python>-config` binary, which is
# not guaranteed to exist for every interpreter (notably the ones provided by
# actions/setup-python). An empty suffix would produce an unimportable module.
PYTHON_EXT := optim_engine$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')" 2>/dev/null)

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

.PHONY: all src benchmarks prelim-tests run-tests test test-python examples python clean help

all: src benchmarks prelim-tests examples python

src: $(SRC_OBJS)

benchmarks: $(BENCH_BIN)

$(BENCH_BIN): $(SRC_OBJS) $(BENCH_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

prelim-tests: $(PRELIM_BINS)

$(BIN_DIR)/prelim-tests/%: $(BUILD_PRELIM)/%.o $(SRC_OBJS) | $(BIN_DIR)/prelim-tests
	$(CXX) $(CXXFLAGS) $^ -o $@

# set -e is essential: without it a failing test binary is reported and then
# ignored, so this target exited 0 even when tests failed.
run-tests: prelim-tests
	@set -e; \
	for t in $(PRELIM_BINS); do \
	    printf '\n--- %s ---\n' "$$t"; \
	    "$$t"; \
	done

# Runs the pybind11 test scripts against the freshly built extension. set -e
# matters here for the same reason it does in run-tests.
test-python: python
	@set -e; \
	for t in tests/python-tests/*.py; do \
	    printf '\n--- %s ---\n' "$$t"; \
	    $(PYTHON) "$$t"; \
	done

# Single gate for CI: C++ suites plus the Python bindings.
test: run-tests test-python

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
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SHARED_LDFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

help:
	@echo ""
	@echo "  all          Build everything (src, benchmarks, prelim-tests, examples, python)"
	@echo "  src          Compile src/ objects only"
	@echo "  benchmarks   Build bin/benchmarks"
	@echo "  prelim-tests Build bin/prelim-tests/<name> for each test"
	@echo "  run-tests    Build and run all prelim tests"
	@echo "  test-python  Build the extension and run the Python binding tests"
	@echo "  test         Everything CI gates on (run-tests + test-python)"
	@echo "  examples     Build bin/<name> for each examples/*.cpp"
	@echo "  python       Build Python extension"
	@echo "  clean        Remove build/ and bin/"
	@echo ""
