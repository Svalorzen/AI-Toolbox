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
introduction to the basics can be found freely online in [this book](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html).

There are many variants of these problems, with single agent worlds, multi agent,
competitive, cooperative, partially observable and so on. This framework is a
work in progress that tries to implement many DTC algorithms in one place, much
like OpenCV is for Computer Vision algorithms.

Please note that the API is not yet stable (although most things at this point
are) since at every algorithm I add I may decide to alter the API a bit, to
offer a more consistent interface throughout the library.

Goals
=====

Decision Theoretic Control is a field which is in rapid development. There are
incredibly many methods to solve problems, each with a huge amounts of
variants. This framework only tries to implement the most influential methods,
and in their vanilla form (or the form that is most widely used in the research
community to my knowledge), trying to keep the code as simple as possible.

If you need any of the variants, the code is structured so that it is easy to
read it and modify it to your requirements, versus providing an endless list of
parameters and include all the variants. Some toolboxes do this, but my opinion
is that this makes the code very hard to digest, which makes it also hard to
find out what parameters to set to get the algorithm variant you want.

Features
========

Single Agent MDP:

- [Value Iteration](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node44.html)
- [Win or Learn Fast Policy Iteration (WoLF)](http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf)
- [Q-Learning](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html)
- [SARSA](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node64.html)
- [Dyna-Q](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node96.html)
- [Prioritized Sweeping](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node98.html)
- [Monte Carlo Tree Search (MCTS)](https://hal.inria.fr/file/index/docid/116992/filename/CG2006.pdf)

Single Agent POMDP:

- [Witness](http://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)
- [Incremental Pruning](http://arxiv.org/pdf/1302.1525.pdf)
- [Point Based Value Iteration (PBVI)](http://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf)
- [POMCP with UCB1](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Applications_files/pomcp.pdf)
- [QMDP](http://dai.fmph.uniba.sk/~petrovic/probrob/ch16.pdf)
- [Real-Time Belief State Search (RTBSS)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.156.2256&rep=rep1&type=pdf)
- [Augmented MDP (AMDP)](http://dai.fmph.uniba.sk/~petrovic/probrob/ch16.pdf)
- [PERSEUS](http://arxiv.org/pdf/1109.2145.pdf)

Python bindings!

Fast Tutorial
=============

In order to use this library you need to have some idea of what a Markov
Decision Process (MDP) is. An MDP is a mathematical framework to work with an
environment which evolves in discrete timestep, and where an agent can influence
its evolution through actions.

A full-on explanation will likely require some math and complicated sounding
terms, so I will explain with an example. The full documentation will include
more details, and currently you can read each class documentation, which helps
in understanding the whole picture.

Suppose you have a 10x10 cell grid world, with an agent in the middle. At each
point, the agent can decide to move up, down, left or right. At any point in
time, you can then describe this world by simply stating where the agent is (for
example, the agent is in cell (5,5)). This description of the world is
absolutely complete and does not require knowledge of the past: it is called a
markovian "state". If knowledge of the past is required, for example to compute
a speed that is then used to move the agent, you simply increase the
dimensionality of the state (add a velocity term to it), until it is markovian
again.

The agent can then influence the environment's state in the next timestep: it
can choose to move, and where it moves will determine the environment next
state. If it moves up, then the next state will be (5,6) with 100% probability.
You can encode this type of movements in a transition table, that for each state
will tell you the probability of ending in another state, given that the agent
performs a certain action.

There is another part of an MDP: reward. Since we want the agent to move in an
intelligent way, we need to tell it what "situations" are better than others.
For example, we may want the agent to move in the top-left corner: thus every
movement the agent does will give him 0 reward, but ending in the top-left
corner will give him 1 reward.

This is all is needed to make this library work. Once you have encoded your
problem in such a way, the code to solve it is generally something like:

    auto model = make_my_model();
    solver_type<decltype(model)> solver( solver_parameters );

    auto solution = solver(model);

Or, for methods that compute the solutions not in one swoop but incrementally at
each timestep the code looks like this:

    auto model = make_my_model();
    solver_type<decltype(model)> solver( model, solver_parameters );
    policy_type policy( solver.getQFunction() );

    for ( unsigned timestep = 0; timestep < max_timestep; ++timestep ) {
        size_t action = policy.sampleAction( current_state );

        std::tie(new_state, reward) = act( action );

        solver.update( current_state, action, new_state, reward);
    }

In particular, states and actions in this library are represented as `size_t`
variables, since for example an (x,y) position can be easily encoded in a single
number.

The code currently in the `example` folder will help you understand the type of
usage (and possibly the `test` folder), and the documentation of each class will
tell you what it is for.

Build Instructions
==================

To build the library you need to install [cmake](http://www.cmake.org/), the
[boost library](http://www.boost.org/) >= 1.53, and the [Eigen 3.2
library](http://eigen.tuxfamily.org/index.php?title=Main_Page). In addition,
C++11 support is required (note: gcc 4.8 will not work as it has a bug which
prevents it from successfully compiling the library, 4.9 will compile everything
correctly). If you want to build the POMDP part of the library, the
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

Compiling a Program
===================

To compile a program that uses this library, simply link it against
`libAIToolboxMDP.a` and possibly both `libAIToolboxPOMDP.a` and all `lp_solve`
libraries. Please note that since the POMDP code relies on the MDP code, you
__MUST__ link the MDP library *after* the POMDP one, otherwise it may result in
`undefined reference` errors.

Documentation
=============

The latest documentation is available [here](http://svalorzen.github.io/AI-Toolbox/).
Keep in mind that it may not always be 100% up to date with the latest
commits, while the one you compile yourself will of course be.
