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
Reinforcement learning focuses on exploring an unknown environment and learning
the best policy for it. An excellent introduction to the basics can be found
freely online in [this
book](http://incompleteideas.net/book/ebook/the-book.html).

There are many variants of these problems, with single agent worlds, multi
agent, multi objective, competitive, cooperative, partially observable and so
on. This framework is a work in progress that tries to implement many DTC
algorithms in one place, much like OpenCV is for Computer Vision algorithms.

Please note that the API may change over time (although most things at this
point are stable) since at every algorithm I add I may decide to alter the API a
bit, to offer a more consistent interface throughout the library.

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

### Cassandra POMDP Format Parsing ###

We parse a reasonable subset of Cassandra's POMDP format, which allows to reuse
already defined problems with this library.

### Python Bindings! ###

Since Python does not allow templates, the classes are binded with as many
as possible instantiations. This toolbox does lose quite a bit of power in
terms of efficient customization when used from Python, but it allows to rapidly
iterate in order to find out what works and what doesn't.

### Bandit/Normal Games: ###

Policies:

- [Greedy Policy](https://arxiv.org/pdf/1707.02038.pdf)
- [Thompson Sampling (Normal distribution)](https://arxiv.org/pdf/1307.3400.pdf)
- [Linear Reward Penalty](https://vtechworks.lib.vt.edu/bitstream/handle/10919/30595/ch3.pdf?sequence=3&isAllowed=y)
- [Exploring Selfish Reinforcement Learning (ESRL)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.102.7547&rep=rep1&type=pdf)

### Single Agent MDP/Stochastic Games: ###

Algorithms:

- [Policy Evaluation](http://incompleteideas.net/book/ebook/node41.html)
- [Policy Iteration](http://incompleteideas.net/book/ebook/node43.html)
- [Value Iteration](http://incompleteideas.net/book/ebook/node44.html)
- [Q-Learning](http://incompleteideas.net/book/ebook/node65.html)
- [Hysteretic Q-Learning](https://hal.archives-ouvertes.fr/hal-00187279/document)
- [SARSA](http://incompleteideas.net/book/ebook/node64.html)
- [Expected SARSA](http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf)
- [SARSA(λ)](http://incompleteideas.net/book/ebook/node77.html)
- [Dyna-Q](http://incompleteideas.net/book/ebook/node96.html)
- [Dyna2](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Applications_files/dyna2.pdf)
- [Prioritized Sweeping](http://incompleteideas.net/book/ebook/node98.html)
- [Monte Carlo Tree Search (MCTS)](https://hal.inria.fr/file/index/docid/116992/filename/CG2006.pdf)

Policies:

- Normal Policy
- Epsilon-Greedy Policy
- Softmax Policy
- Q-Greedy Policy
- [Win or Learn Fast Policy Iteration (WoLF)](http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf)

### Single Agent POMDP: ###

Algorithms:

- [Augmented MDP (AMDP)](http://dai.fmph.uniba.sk/~petrovic/probrob/ch16.pdf)
- [Blind Strategies](http://www.aaai.org/Papers/AAAI/1997/AAAI97-114.pdf)
- [Fast Informed Bound](https://people.cs.pitt.edu/~milos/research/JAIR-2000.pdf)
- [GapMin](https://cs.uwaterloo.ca/~ppoupart/publications/gapMin/gap-camera-ready.pdf)
- [Incremental Pruning](http://arxiv.org/pdf/1302.1525.pdf)
- [PERSEUS](http://arxiv.org/pdf/1109.2145.pdf)
- [Point Based Value Iteration (PBVI)](http://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf)
- [POMCP with UCB1](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Applications_files/pomcp.pdf)
- [QMDP](http://dai.fmph.uniba.sk/~petrovic/probrob/ch16.pdf)
- [rPOMCP](https://esc.fnwi.uva.nl/thesis/centraal/files/f581932172.pdf)
- [Real-Time Belief State Search (RTBSS)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.156.2256&rep=rep1&type=pdf)
- [Witness](http://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)

Policies:

- Normal Policy

### Factored/Joined Multi-Agent: ###

#### Bandits: ####

Not in Python yet.

Algorithms:

- [Variable Elimination](https://papers.nips.cc/paper/1941-multiagent-planning-with-factored-mdps.pdf)
- [Multi-Objective Variable Elimination](https://staff.fnwi.uva.nl/s.a.whiteson/pubs/roijersaamas13.pdf)
- [Learning with Linear Rewards (LLR)](https://arxiv.org/pdf/1011.4748.pdf)

Policies:

- Q-Greedy Policy

#### MDP: ####

Not in Python yet.

Algorithms:

- [Sparse Cooperative QLearning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.99.8394&rep=rep1&type=pdf)
- [Joint Action Learners](https://www.aaai.org/Papers/AAAI/1998/AAAI98-106.pdf)
- [FactoredLP](https://ai.stanford.edu/~koller/Papers/Guestrin+al:IJCAI01.pdf)

Policies:

- SingleAction Policy
- Epsilon-Greedy Policy
- Q-Greedy Policy

Build Instructions
==================

Dependencies
------------

To build the library you need:

- [cmake](http://www.cmake.org/) >= 3.9
- the [boost library](http://www.boost.org/) >= 1.62
- the [Eigen 3.3 library](http://eigen.tuxfamily.org/index.php?title=Main_Page).

In addition, full C++17 support is now required (**this means at least g++-7**)

If you want to build the POMDP or Factored/Multi-Agent parts of the library you
will also need:

- the [lp\_solve library](http://lpsolve.sourceforge.net/5.5/) (a shared library
  must be available to compile the Python wrapper).

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
MAKE_LIB         # Builds the whole core C++ libraries (MDP, POMDP, etc..)
MAKE_MDP         # Builds only the core C++ MDP library
MAKE_FMDP        # Builds only the core C++ Factored/Multi-Agent and MDP libraries
MAKE_POMDP       # Builds only the core C++ POMDP and MDP libraries
MAKE_PYTHON      # Builds Python bindings for the compiled core libraries
MAKE_TESTS       # Builds the library's tests for the compiled core libraries
MAKE_EXAMPLES    # Builds the library's examples using the compiled core libraries
```

These flags can be combined as needed. For example:

```bash
# Will build MDP and MDP Python bindings
cmake -DCMAKE_BUILD_TYPE=Debug -DMAKE_MDP=1 -DMAKE_PYTHON=1 ..
```

The default flags when nothing is specified are `MAKE_ALL` and
`CMAKE_BUILD_TYPE=Release`.

The static library files will be available directly in the build directory.
Three separate libraries are built: `AIToolboxMDP`, `AIToolboxPOMDP` and
`AIToolboxFMDP`. In case you want to link against either the POMDP library or
the Factored MDP library, you will also need to link against the MDP one, since
both of them use MDP functionality.

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

To compile a program that uses this library, simply link it against the compiled
libraries you need, and possibly to the `lp_solve` libraries (if using POMDP or
FMDP).

Please note that since both POMDP and FMDP libraries rely on the MDP code, you
__MUST__ specify those libraries *before* the MDP library when linking,
otherwise it may result in `undefined reference` errors. The POMDP and Factored
MDP libraries are not currently dependent on each other so their order does not
matter.

For Python, you just need to import the `AIToolbox.so` module, and you'll be
able to use the classes as exported to Python. All classes are documented, and
you can run in the Python CLI

    help(AIToolbox.MDP)
    help(AIToolbox.POMDP)

to see the documentation for each specific class.

Documentation
=============

The latest documentation is available [here](http://svalorzen.github.io/AI-Toolbox/).
Keep in mind that it may not always be 100% up to date with the latest
commits, while the one you compile yourself will of course be.

For Python docs you can find them by typing `help(AIToolbox)` from the
interpreter. It should show the exported API for each class, along with any
differences in input/output.
