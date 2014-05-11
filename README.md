AI-Toolbox [![Build Status](https://travis-ci.org/Svalorzen/AI-Toolbox.svg?branch=master)](https://travis-ci.org/Svalorzen/AI-Toolbox) [![Stories in Ready](https://badge.waffle.io/Svalorzen/AI-Toolbox.png?label=ready)](https://waffle.io/Svalorzen/AI-Toolbox)
==========

This C++ toolbox is aimed at representing and solving common AI problems,
implementing an easy-to-use interface which should be hopefully extensible
to many problems, while keeping code readable.

Current development includes MDPs, POMDPs and related algorithms. This toolbox
has been developed taking inspiration from the Matlab MDPToolbox,
which you can find [here](http://www7.inra.fr/mia/T/MDPtoolbox/).

Brief
=====

This toolbox is aimed at Decision Theoretic Control algorithms. The general idea
is to create algorithms that are able to interact with an environment in order
to obtain some reward using actions, and to find the best policy of actions to
use to do so.

The field divides itself into planning and learning: planning focuses into
solving problems that we know how to model: think chess. Learning focuses into
creating a model for an environment we do not know in advance, and subsequently
learn the best policy for it.

There are many variants of these problems, with single agent world, multi agent,
competitive, cooperative, partially observable and so on. This framework is a
work in progress that tries to implement many DTC algorithms in one place, much
like OpenCV is for Computer Vision algorithms.

Currently the available functionality is very little, and the API is not yet
stable, as I can only work on this inbetween my thesis, and I generally only
have time to insert algorithms that I happen to use for other projects too. My
hope is that one day it will be big enough that I will be able to publicize it
and hopefully obtain contribution from other scientists that want to use these
type of methods.

Build Instructions
==================

To build the library you need to install [cmake](http://www.cmake.org/) and
the [boost library](http://www.boost.org/) >= 1.53. In addition, C++11 support
is required. If you want to build the POMDP part of the library, the
[lp\_solve](http://lpsolve.sourceforge.net/5.5/) library is also required.

After that, you can simply execute the following commands from the project's
main folder:

```bash
mkdir build
cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

The static library files will be available directly in the `build` directory. At
the moment two separate libraries are created: `AIToolboxMDP` and
`AIToolboxPOMDP`. In case you want to link against the POMDP library, you will
also need to link against the MDP one, since POMDP uses MDP functionality.

In case you do not want to build the whole library (due for example to the
lp\_solve requirements) you may specify to cmake what parts of the library you
actually want to build, like so:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DMAKE_MDP=1 ..   # Will only build the MDP algorithms
cmake -DCMAKE_BUILD_TYPE=Release -DMAKE_POMDP=1 .. # Will build both MDP and POMDP algorithms
```

A number of small tests are included which you can find in the `test/` folder.
You can execute them after building the project using the following command
directly from the `build` directory, just after you finish `make`:

```bash
ctest
```

The tests also offer a brief introduction for the framework, waiting for a
more complete descriptive write-up. Only the tests for the parts of the library
that you compiled are going to be built.

To compile the library's documentation you need the [Doxygen](http://www.stack.nl/~dimitri/doxygen/)
tool. To use it it is sufficient to execute the following command from the
project's main folder:

```bash
doxygen
```

After that the documentation will be generated into an `html` folder in the
main directory.

Documentation
=============

The latest documentation is available [here](http://svalorzen.github.io/AI-Toolbox/).
Keep in mind that it may not always be 100% up to date with the latest
commits, while the one you compile yourself will of course be.

