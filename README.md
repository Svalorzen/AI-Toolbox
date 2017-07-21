AI-Toolbox [![Build Status](https://travis-ci.org/Svalorzen/AI-Toolbox.svg?branch=master)](https://travis-ci.org/Svalorzen/AI-Toolbox) [![Stories in Ready](https://badge.waffle.io/Svalorzen/AI-Toolbox.png?label=ready)](https://waffle.io/Svalorzen/AI-Toolbox)
==========

This C++ toolbox is aimed at representing and solving common AI problems,
implementing an easy-to-use interface which should be hopefully extensible
to many problems, while keeping code readable.

Current development includes MDPs, POMDPs and related algorithms. This toolbox
has been developed taking inspiration from the Matlab `MDPToolbox`, which you
can find [here](http://www7.inra.fr/mia/T/MDPtoolbox/), and from the
`pomdp-solve` software written by A. R. Cassandra, which you can find
[here](http://www.pomdp.org/code/index.shtml).

Brief
=====

This toolbox is aimed at Decision Theoretic Control algorithms. The general idea
is to create algorithms that are able to interact with an environment in order
to obtain some reward using actions, and to find the best policy of actions to
use to do so.

The field divides itself into planning and reinforcement learning: planning
focuses into solving problems that we know how to model: think chess, or 2048.
Reinforcement learning focuses into creating a model for an environment we do
not know in advance, and while learning the best policy for it. An excellent
introduction to the basics can be found freely online in [this
book](http://incompleteideas.net/sutton/book/ebook/the-book.html).

There are many variants of these problems, with single agent worlds, multi
agent, multi objective, competitive, cooperative, partially observable and so
on. This framework is a work in progress that tries to implement many DTC
algorithms in one place, much like OpenCV is for Computer Vision algorithms.

Please note that the API is not yet stable (although most things at this point
are) since at every algorithm I add I may decide to alter the API a bit, to
offer a more consistent interface throughout the library.

Goals
=====

Decision Theoretic Control is a field which is in rapid development. There are
incredibly many methods to solve problems, each with a huge number of variants.
This framework only tries to implement the most influential methods, and in
their vanilla form (or the form that is most widely used in the research
community to my knowledge), trying to keep the code as simple as possible.

If you need any of the variants, the code is structured so that it is easy to
read it and modify it to your requirements, versus providing an endless list of
parameters and include all the variants. Some toolboxes do this, but my opinion
is that this makes the code very hard to digest, which makes it also hard to
find out what parameters to set to get the algorithm variant you want.

Features
========

### Python bindings! ###

Since Python does not allow templates, the classes are binded with as many
as possible instantiations. This toolbox does lose quite a bit of power in
terms of efficient customization when used from Python, but it allows to rapidly
iterate in order to find out what works and what doesn't.

### Single Agent MDP: ###

Algorithms:

- [Policy Evaluation](http://incompleteideas.net/sutton/book/ebook/node41.html)
- [Policy Iteration](http://incompleteideas.net/sutton/book/ebook/node43.html)
- [Value Iteration](http://incompleteideas.net/sutton/book/ebook/node44.html)
- [Win or Learn Fast Policy Iteration (WoLF)](http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf)
- [Q-Learning](http://incompleteideas.net/sutton/book/ebook/node65.html)
- [SARSA](http://incompleteideas.net/sutton/book/ebook/node64.html)
- [Dyna-Q](http://incompleteideas.net/sutton/book/ebook/node96.html)
- [Prioritized Sweeping](http://incompleteideas.net/sutton/book/ebook/node98.html)
- [Monte Carlo Tree Search (MCTS)](https://hal.inria.fr/file/index/docid/116992/filename/CG2006.pdf)

Policies:

- Normal Policy
- Epsilon-Greedy Policy
- Softmax Policy
- Q-Greedy Policy
- [WoLF Policy](http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf)

### Single Agent POMDP: ###

Algorithms:

- [Witness](http://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)
- [Incremental Pruning](http://arxiv.org/pdf/1302.1525.pdf)
- [Point Based Value Iteration (PBVI)](http://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf)
- [POMCP with UCB1](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Applications_files/pomcp.pdf)
- [QMDP](http://dai.fmph.uniba.sk/~petrovic/probrob/ch16.pdf)
- [Real-Time Belief State Search (RTBSS)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.156.2256&rep=rep1&type=pdf)
- [Augmented MDP (AMDP)](http://dai.fmph.uniba.sk/~petrovic/probrob/ch16.pdf)
- [PERSEUS](http://arxiv.org/pdf/1109.2145.pdf)

Policies:

- Normal Policy

### Factored/Multi-Agent MDP: ###

Not in Python yet.

Algorithms:

- [Variable Elimination](http://www.select.cs.cmu.edu/publications/paperdir/nips2001-guestrin-koller-parr.pdf)
- [Multi-Objective Variable Elimination](https://staff.fnwi.uva.nl/s.a.whiteson/pubs/roijersaamas13.pdf)
- [Sparse Cooperative QLearning](http://www.machinelearning.org/proceedings/icml2004/papers/267.pdf)
- [Learning with Linear Rewards (LLR)](http://www-scf.usc.edu/~ygai/publications/TON2012.pdf)

Policies:

- SingleAction Policy
- Epsilon-Greedy Policy
- Q-Greedy Policy

Build Instructions
==================

Dependencies
------------

To build the library you need:

- [cmake](http://www.cmake.org/)
- the [boost library](http://www.boost.org/) >= 1.62
- the [Eigen 3.3 library](http://eigen.tuxfamily.org/index.php?title=Main_Page).

In addition, full C++17 support is now required (**this means at least g++-7**)

If you want to build the POMDP part of the library you will also need:

- the [lp\_solve](http://lpsolve.sourceforge.net/5.5/) library is also required
  (a shared library must be available to compile the Python libraries).

Building
--------

Once you have all required dependencies, you can simply execute the following
commands from the project's main folder:

```bash
mkdir build
cd build/
cmake ..
make
```

`cmake` can be called with a series of flags in order to customize the output,
if building everything is not desirable. The following flags are available:

```bash
CMAKE_BUILD_TYPE # Defines the build type
MAKE_ALL         # Builds all there is to build in the project
MAKE_LIB         # Builds the core C++ library
MAKE_MDP         # Builds the core C++ MDP library
MAKE_FMDP        # Builds the core C++ Factored MDP and MDP library
MAKE_POMDP       # Builds the core C++ POMDP and MDP library
MAKE_PYTHON      # Builds Python bindings for the compiled core library
MAKE_TESTS       # Builds the library's tests for the compiled core library
MAKE_EXAMPLES    # Builds the library's examples using the compiled core library
```

These flags can be combined as needed. For example:

```bash
# Will build MDP and MDP Python bindings
cmake -DCMAKE_BUILD_TYPE=Debug -DMAKE_MDP=1 -DMAKE_PYTHON=1 ..
```

The default flags when nothing is specified are `MAKE_ALL` and
`CMAKE_BUILD_TYPE=Release`.

The static library files will be available directly in the build directory. At
the moment two separate libraries are created: `AIToolboxMDP` and
`AIToolboxPOMDP`. In case you want to link against the POMDP library, you will
also need to link against the MDP one, since POMDP uses MDP functionality.

A number of small tests are included which you can find in the `test/` folder.
You can execute them after building the project using the following command
directly from the `build` directory, just after you finish `make`:

```bash
ctest
```

The tests also offer a brief introduction for the framework, waiting for a
more complete descriptive write-up. Only the tests for the parts of the library
that you compiled are going to be built.

To compile the library's documentation you need the
[Doxygen](http://www.stack.nl/~dimitri/doxygen/) tool. To use it it is
sufficient to execute the following command from the project's root folder:

```bash
doxygen
```

After that the documentation will be generated into an `html` folder in the
main directory.

Compiling a Program
===================

To compile a program that uses this library, simply link it against
`libAIToolboxMDP.a` and possibly both `libAIToolboxPOMDP.a` and all `lp_solve`
libraries. Please note that since the POMDP code relies on the MDP code, you
__MUST__ link the MDP library *after* the POMDP one, otherwise it may result in
`undefined reference` errors. The same is true for the MDP and Factored MDP
libraries. The POMDP and Factored MDP libraries are not currently dependent so
their order does not matter.

For Python, you just need to import the `MDP.so` and `POMDP.so` modules, and
you'll be able to use the classes as exported to Python. All classes are
documented, and you can run in the Python CLI

    help(MDP)
    help(POMDP)

to see the documentation for each specific class.

Documentation
=============

The latest documentation is available [here](http://svalorzen.github.io/AI-Toolbox/).
Keep in mind that it may not always be 100% up to date with the latest
commits, while the one you compile yourself will of course be.

For Python docs you can find them by typing `help(MDP)` or `help(MDP.SomeMDPClass)`
from the interpreter. It should show the exported API for each class, along with
any differences in input/output.
