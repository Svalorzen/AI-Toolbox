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
RUN cd /usr/src && git clone https://github.com/Svalorzen/AI-Toolbox.git 

# working directory (for successive commands)
WORKDIR /usr/src/AI-Toolbox

# Create build directory
RUN mkdir build

# Export CXX compiler path
RUN export CXX=/usr/bin/g++-10

# Also tested:
# (✓) cmake -DCMAKE_CXX_COMPILER=$CXX -DMAKE_MDP=1 ..
# (✓) cmake -DCMAKE_CXX_COMPILER=$CXX -DMAKE_MDP=1 -DMAKE_POMDP=1 ..
# (✓) cmake -DCMAKE_CXX_COMPILER=$CXX -DMAKE_MDP=1 -DMAKE_POMDP=1 -DMAKE_EXAMPLES=1 ..

RUN cd build && \
 cmake  -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 \
        -DMAKE_MDP=1 \
        -DMAKE_POMDP=1 \
        -DMAKE_EXAMPLES=1 \
        -DMAKE_PYTHON=1 \
        -DAI_PYTHON_VERSION=3 /usr/src/AI-Toolbox

# Build AI-Toolbox
RUN cd build && make

# Run examples
RUN cd build/examples && ./cliff QL

RUN cd build && cp AIToolbox.so examples/
RUN cd build/examples && python3 ./tiger_antelope.py

#### OUTPUT: ####
# -- The CXX compiler identification is GNU 10.5.0
# -- Check for working CXX compiler: /usr/bin/g++-10    
# -- Check for working CXX compiler: /usr/bin/g++-10 -- works
# -- Detecting CXX compiler ABI info
# -- Detecting CXX compiler ABI info - done
# -- Detecting CXX compile features
# -- Detecting CXX compile features - done
# -- Found Boost: /usr/lib/aarch64-linux-gnu/cmake/Boost-1.71.0/BoostConfig.cmake (found suitable version "1.71.0", minimum required is "1.67")  
# -- Found Eigen3: /usr/include/eigen3 (Required is at least version "3.2.92") 
# -- Performing Test LPSOLVE_LINKS_ALONE
# -- Performing Test LPSOLVE_LINKS_ALONE - Success
# -- Found LpSolve: /usr/lib/lp_solve/liblpsolve55.so  
# -- Found Python: /usr/bin/python3.8 (found version "3.8.10") found components: Interpreter Development 
# -- Found Boost: /usr/lib/aarch64-linux-gnu/cmake/Boost-1.71.0/BoostConfig.cmake (found suitable version "1.71.0", minimum required is "1.67") found components: python38 

# ### AI_TOOLBOX SETTINGS ###
# -- CMAKE_BUILD_TYPE:   Release
# -- IPO / LTO enabled
# ✗ NOT Building everything      (-DMAKE_ALL=0)
# ✗ NOT Building entire library  (-DMAKE_LIB=0)
# ✓ Building MDP             (-DMAKE_MDP=1)
# ✗ NOT Building Factored MDP    (-DMAKE_FMDP=0)
# ✓ Building POMDP           (-DMAKE_POMDP=1)
# ✗ NOT Building Tests           (-DMAKE_TESTS=0)
# ✓ Building Examples        (-DMAKE_EXAMPLES=1)
# ✓ Building Python bindings (-DMAKE_PYTHON=1)
#   - Selected Python 3.8    (-DAI_PYTHON_VERSION=3)
# ✗ NOT Enabled runtime logging  (-DAI_LOGGING_ENABLED=0)

# -- Configuring done
# -- Generating done
# -- Build files have been written to: /usr/src/AI-Toolbox/build