# Reproducible build and test environment for the numerical optimization engine.
#
#   docker build -t optim-engine .
#   docker run --rm optim-engine                       # C++ suites + Python bindings
#   docker build --build-arg CXX=clang++-18 .
#
# Base image pinned by digest so this builds the same way later.
FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ARG CXX=g++-14
ENV CXX=${CXX}

# libeigen3-dev provides the pkg-config file the Makefile discovers; python3-dev
# provides Python.h, which pybind11 needs to compile the extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++-14 \
        clang-18 \
        make \
        pkg-config \
        libeigen3-dev \
        python3 \
        python3-dev \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir pybind11 numpy

WORKDIR /src
COPY . .

RUN make -j"$(nproc)"

CMD ["make", "test"]
