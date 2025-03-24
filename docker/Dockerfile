## Dockerfile for AI-Toolbox
# To build the docker image, run: docker build . -t ai-toolbox-full:0

# Docker image
FROM ubuntu:focal-20231211

RUN apt-get update

RUN apt-get install -y g++-10
RUN apt-get install -y cmake 
RUN apt-get install -y liblpsolve55-dev
RUN apt-get install -y lp-solve
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y git
RUN apt-get install -y rsync
RUN apt-get install -y python3-pip

# clone AI-Toolbox GitHub repository
RUN cd /usr/src && git clone --branch docker-image https://github.com/omniscientoctopus/AI-Toolbox.git

# working directory (for successive commands)
WORKDIR /usr/src/AI-Toolbox

# Create build directory
RUN mkdir build

# Export CXX compiler path
RUN export CXX=/usr/bin/g++-10

# Configure AI-Toolbox
RUN cd build && \
 cmake  -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 \
        -DMAKE_ALL=1 \
        -DMAKE_PYTHON=1 \
        -DAI_PYTHON_VERSION=3\
        -DAI_LOGGING_ENABLED=1 /usr/src/AI-Toolbox

# Build AI-Toolbox
RUN cd build && make

# Run examples

# Run C++ example
RUN cd build/examples && ./cliff QL

# Run Python example
RUN cd build && cp AIToolbox.so examples/
RUN cd build/examples && python3 ./tiger_antelope.py

# Check which MDP/POMDP modules are available
RUN cd build/examples && python3 -c "import AIToolbox; help(AIToolbox.MDP)"

RUN cd build/examples && python3 -c "import AIToolbox; help(AIToolbox.POMDP)"

# Test
# continue even if tests fail
#RUN cd build && ctest -V -j2 --output-on-failure --repeat until-pass:5
