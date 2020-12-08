AI-Toolbox
==========

[![Library overview video](https://user-images.githubusercontent.com/1609228/99919181-3404dc00-2d1c-11eb-8593-0bf6af44cef8.png)](https://www.youtube.com/watch?v=qjSo41DVSXg)

[![Build Status](https://github.com/Svalorzen/AI-Toolbox/workflows/CMake/badge.svg?branch=master)](https://github.com/Svalorzen/AI-Toolbox/actions?query=workflow%3A"CMake")


This C++ toolbox is aimed at representing and solving common AI problems,
implementing an easy-to-use interface which should be hopefully extensible
to many problems, while keeping code readable.

Current development includes MDPs, POMDPs and related algorithms. This toolbox
was originally developed taking inspiration from the Matlab `MDPToolbox`, which
you can find [here](https://miat.inrae.fr/MDPtoolbox/), and from the
`pomdp-solve` software written by A. R. Cassandra, which you can find
[here](http://www.pomdp.org/code/index.shtml).

If you use this toolbox for research, please consider citing our JMLR article:

```
@article{JMLR:v21:18-402,
  author  = {Eugenio Bargiacchi and Diederik M. Roijers and Ann Now\'{e}},
  title   = {AI-Toolbox: A C++ library for Reinforcement Learning and Planning (with Python Bindings)},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {102},
  pages   = {1-12},
  url     = {http://jmlr.org/papers/v21/18-402.html}
}
```

Description
===========

This toolbox provides implementations of several reinforcement learning (RL)
and planning algorithms. An excellent introduction to the basics can be found
freely online in [this book](http://incompleteideas.net/book/ebook/the-book.html).

The implemented algorithms can be applied in several settings: single agent
environments, multi agent, multi objective, competitive, cooperative, partially
observable and so on. We strive to maintain a consistent interface throughout all
domains for ease of use. The toolbox is actively developed and used in research.

Implementations are kept as simple as possible and with relatively few options
compared to other libraries; we believe that this makes the code easier to read
and modify to best suit your needs.

Please note that the API may change over time (although most things at this
point are stable) since as the toolbox grows I may decide to alter it to improve
overall consistency.

Documentation
=============

The latest documentation is available [here](http://svalorzen.github.io/AI-Toolbox/).
Keep in mind that it may not always be 100% up to date with the latest
commits, while the one you compile yourself will of course be.

For Python docs you can find them by typing `help(AIToolbox)` from the
interpreter. It should show the exported API for each class, along with any
differences in input/output.

Features
========

### Cassandra POMDP Format Parsing ###

Cassandra's POMDP format is a type of text file that contains a definition of an
MDP or POMDP model. You can find some examples
[here](http://pomdp.org/examples/). While it is absolutely not necessary to use
this format, and you can define models via code, we do parse a reasonable subset
of Cassandra's POMDP format, which allows to reuse already defined problems with
this library. [Here's the docs on that](http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Impl_1_1CassandraParser.html).

### Python 2 and 3 Bindings! ###

The user interface of the library is pretty much the same with Python than what
you would get by using simply C++. See the `examples` folder to see just how
much Python and C++ code resemble each other. Since Python does not allow
templates, the classes are binded with as many instantiations as possible.

Additionally, the library allows the usage of native Python generative models
(where you don't need to specify the transition and reward functions, you only
sample next state and reward). This allows for example to directly use OpenAI
gym environments with minimal code writing.

That said, if you need to customize a specific implementation to make it perform
better on your specific use-cases, or if you want to try something completely
new, you will have to use C++.

### Utilities ###

The library has an extensive set of utilities which would be too long to
enumerate here. In particular, we have utilities for [combinatorics][1],
[polytopes][2], [linear programming][3], [sampling and distributions][4],
[automated statistics][5], [belief updating][6], [many][7] [data][8] [structures][9],
[logging][10], [seeding][11] and much more.

[1]: http://svalorzen.github.io/AI-Toolbox/Combinatorics_8hpp.html
[2]: http://svalorzen.github.io/AI-Toolbox/Polytope_8hpp.html
[3]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1LP.html
[4]: http://svalorzen.github.io/AI-Toolbox/Probability_8hpp.html
[5]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Statistics.html
[6]: http://svalorzen.github.io/AI-Toolbox/include_2AIToolbox_2POMDP_2Utils_8hpp.html
[7]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Trie.html
[8]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1FactorGraph.html
[9]: http://svalorzen.github.io/AI-Toolbox/FactoredMatrix_8hpp.html
[10]: http://svalorzen.github.io/AI-Toolbox/logging.html
[11]: http://svalorzen.github.io/AI-Toolbox/Seeder_8hpp.html

### Bandit/Normal Games: ###

|                                                       | **Policies**                                     |                      |
| :---------------------------------------------------: | :----------------------------------------------: | :------------------: |
| [Exploring Selfish Reinforcement Learning (ESRL)][12] | [Q-Greedy Policy][13]                            | [Softmax Policy][14] |
| [Linear Reward Penalty][15]                           | [Thompson Sampling (Student-t distribution)][16] | [Random Policy][17]  |

[12]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1ESRLPolicy.html "Exploring selfish reinforcement learning in repeated games with stochastic rewards, Verbeeck et al."
[13]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1QGreedyPolicy.html "A Tutorial on Thompson Sampling, Russo et al."
[14]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1QSoftmaxPolicy.html "Reinforcement Learning: An Introduction, Ch 2.3, Sutton & Barto"
[15]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1LRPPolicy.html "Self-organization in large populations of mobile robots, Ch 3: Stochastic Learning Automata, Unsal"
[16]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1ThompsonSamplingPolicy.html "Thompson Sampling for 1-Dimensional Exponential Family Bandits, Korda et al."
[17]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1RandomPolicy.html

### Single Agent MDP/Stochastic Games: ###

|                                       | **Models**                                                 |                                                 |
| :-----------------------------------: | :--------------------------------------------------------: | :---------------------------------------------: |
| [Basic Model][18]                     | [Sparse Model][19]                                         | [Maximum Likelihood Model][20]                  |
| [Sparse Maximum Likelihood Model][21] | [Thompson Model (Dirichlet + Student-t distributions)][22] |                                                 |
|                                       | **Algorithms**                                             |                                                 |
| [Dyna-Q][23]                          | [Dyna2][24]                                                | [Expected SARSA][25]                            |
| [Hysteretic Q-Learning][26]           | [Importance Sampling][27]                                  | [Linear Programming][28]                        |
| [Monte Carlo Tree Search (MCTS)][29]  | [Policy Evaluation][30]                                    | [Policy Iteration][31]                          |
| [Prioritized Sweeping][32]            | [Q-Learning][33]                                           | [Double Q-Learning][34]                         |
| [Q(λ)][35]                            | [R-Learning][36]                                           | [SARSA(λ)][37]                                  |
| [SARSA][38]                           | [Retrace(λ)][39]                                           | [Tree Backup(λ)][40]                            |
| [Value Iteration][41]                 |                                                            |                                                 |
|                                       | **Policies**                                               |                                                 |
| [Basic Policy][42]                    | [Epsilon-Greedy Policy][43]                                | [Softmax Policy][44]                            |
| [Q-Greedy Policy][45]                 | [PGA-APP][46]                                              | [Win or Learn Fast Policy Iteration (WoLF)][47] |

[18]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1Model.html
[19]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1SparseModel.html
[20]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1MaximumLikelihoodModel.html
[21]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1SparseMaximumLikelihoodModel.html
[22]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1ThompsonModel.html

[23]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1DynaQ.html, "Reinforcement Learning: An Introduction, Ch 9.2, Sutton & Barto"
[24]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1Dyna2.html "Sample-Based Learning and Search with Permanent and Transient Memories, Silver et al."
[25]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1ExpectedSARSA.html "A Theoretical and Empirical Analysis of Expected Sarsa, van Seijen et al."
[26]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1HystereticQLearning.html "Hysteretic Q-Learning : an algorithm for decentralized reinforcement learning in cooperative multi-agent teams, Matignon et al."
[27]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1ImportanceSampling.html "Eligibility Traces for Off-Policy Policy Evaluation, Precup"
[28]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1LinearProgramming.html
[29]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1MCTS.html "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search, Coulom"
[30]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1PolicyEvaluation.html "Reinforcement Learning: An Introduction, Ch 4.1, Sutton & Barto"
[31]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1PolicyIteration.html "Reinforcement Learning: An Introduction, Ch 4.3, Sutton & Barto"
[32]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1PrioritizedSweeping.html "Reinforcement Learning: An Introduction, Ch 9.4, Sutton & Barto"
[33]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1QLearning.html "Reinforcement Learning: An Introduction, Ch 6.5, Sutton & Barto"
[34]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1DoubleQLearning.html "Double Q-learning, van Hasselt"
[35]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1QL.html "Q(λ) with Off-Policy Corrections, Harutyunyan et al."
[36]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1RLearning.html "A Reinforcement Learning Method for Maximizing Undiscounted Rewards, Schwartz"
[37]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1SARSAL.html "Reinforcement Learning: An Introduction, Ch 7.5, Sutton & Barto"
[38]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1SARSA.html "Reinforcement Learning: An Introduction, Ch 6.4, Sutton & Barto"
[39]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1RetraceL.html "Safe and efficient off-policy reinforcement learning, Munos et al."
[40]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1TreeBackupL.html "Eligibility Traces for Off-Policy Policy Evaluation, Precup"
[41]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1ValueIteration.html "Reinforcement Learning: An Introduction, Ch 4.4, Sutton & Barto"

[42]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1Policy.html
[43]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1EpsilonPolicy.html
[44]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1QSoftmaxPolicy.html
[45]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1QGreedyPolicy.html
[46]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1PGAAPPPolicy.html "Multi-Agent Learning with Policy Prediction, Zhang et al."
[47]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1WoLFPolicy.html "Rational and Convergent Learning in Stochastic Games, Bowling et al."

### Single Agent POMDP: ###

|                            | **Models**                                  |                                          |
| :------------------------: | :-----------------------------------------: | :--------------------------------------: |
| [Basic Model][48]          | [Sparse Model][49]                          |                                          |
|                            | **Algorithms**                              |                                          |
| [Augmented MDP (AMDP)][50] | [Blind Strategies][51]                      | [Fast Informed Bound][52]                |
| [GapMin][53]               | [Incremental Pruning][54]                   | [Linear Support][55]                     |
| [PERSEUS][56]              | [POMCP with UCB1][57]                       | [Point Based Value Iteration (PBVI)][58] |
| [QMDP][59]                 | [Real-Time Belief State Search (RTBSS)][60] | [SARSOP][61]                             |
| [Witness][62]              | [rPOMCP][63]                                |                                          |
|                            | **Policies**                                |                                          |
| [Basic Policy][64]         |                                             |                                          |

[48]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1Model.html
[49]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1SparseModel.html

[50]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1AMDP.html "Probabilistic robotics, Ch 16: Approximate POMDP Techniques, Thrun"
[51]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1BlindStrategies.html "Incremental methods for computing bounds in partially observable Markov decision processes, Hauskrecht"
[52]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1FastInformedBound.html "Value-Function Approximations for Partially Observable Markov Decision Processes, Hauskrecht"
[53]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1GapMin.html "Closing the Gap: Improved Bounds on Optimal POMDP Solutions, Poupart et al."
[54]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1IncrementalPruning.html "Incremental Pruning: A Simple, Fast, Exact Method for Partially Observable Markov Decision Processes, Cassandra et al."
[55]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1LinearSupport.html "Algorithms for Partially Observable Markov Decision Processes, Phd Thesis, Cheng"
[56]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1PERSEUS.html "Perseus: Randomized Point-based Value Iteration for POMDPs, Spaan et al."
[57]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1POMCP.html "Monte-Carlo Planning in Large POMDPs, Silver et al."
[58]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1PBVI.html "Point-based value iteration: An anytime algorithm for POMDPs, Pineau et al."
[59]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1QMDP.html "Probabilistic robotics, Ch 16: Approximate POMDP Techniques, Thrun"
[60]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1RTBSS.html "Real-Time Decision Making for Large POMDPs, Paquet et al."
[61]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1SARSOP.html "SARSOP: Efficient Point-Based POMDP Planning by Approximating Optimally Reachable Belief Spaces, Kurniawati et al."
[62]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1Witness.html "Planning and acting in partially observable stochastic domains, Kaelbling et al."
[63]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1rPOMCP.html "Dynamic Resource Allocation for Multi-Camera Systems, Phd Thesis, Bargiacchi"

[64]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1Policy.html

### Factored/Joint Multi-Agent: ###

#### Bandits: ####

Not in Python yet.

|                                                        | **Algorithms**                                               |                                                    |
| :----------------------------------------------------: | :----------------------------------------------------------: | :------------------------------------------------: |
| [Max-Plus][65]                                         | [Multi-Objective Variable Elimination (MOVE)][66]            | [Upper Confidence Variable Elimination (UCVE)][67] |
| [Variable Elimination][68]                             |                                                              |                                                    |
|                                                        | **Policies**                                                 |                                                    |
| [Q-Greedy Policy][69]                                  | [Random Policy][70]                                          | [Learning with Linear Rewards (LLR)][71]           |
| [Multi-Agent Upper Confidence Exploration (MAUCE)][72] | [Multi-Agent Thompson-Sampling (Student-t distribution)][73] | [Single-Action Policy][74]                         |

[65]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1MaxPlus.html "Collaborative Multiagent Reinforcement Learning by Payoff Propagation, Kok et al."
[66]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1MultiObjectiveVariableElimination.html "Multi-Objective Variable Elimination for Collaborative Graphical Games, Roijers et al."
[67]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1UCVE.html "Learning to Coordinate with Coordination Graphs in Repeated Single-Stage Multi-Agent Decision Problems, Bargiacchi et al."
[68]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1VariableElimination.html "Multiagent Planning with Factored MDPs, Guestrin et al."

[69]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1QGreedyPolicy.html
[70]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1RandomPolicy.html
[71]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1LLRPolicy.html "Combinatorial Network Optimization with Unknown Variables: Multi-Armed Bandits with Linear Rewards, Gai et al."
[72]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1MAUCEPolicy.html "Learning to Coordinate with Coordination Graphs in Repeated Single-Stage Multi-Agent Decision Problems, Bargiacchi et al."
[73]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1ThompsonSamplingPolicy.html "Multi-Agent Thompson Sampling for Bandit Applications with Sparse Neighbourhood Structures, Verstraeten et al."
[74]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1SingleActionPolicy.html

#### MDP: ####

Not in Python yet.

|                                     | **Models**                                 |                                                                        |
| :---------------------------:       | :----------------------------------------: | :--------------------------------------------------------------------: |
| [Cooperative Basic Model][75]       | [Cooperative Maximum Likelihood Model][76] | [Cooperative Thompson Model (Dirichlet + Student-t distributions)][77] |
|                                     | **Algorithms**                             |                                                                        |
| [FactoredLP][78]                    | [Multi Agent Linear Programming][79]       | [Joint Action Learners][80]                                            |
| [Sparse Cooperative Q-Learning][81] | [Cooperative Prioritized Sweeping][82]     |                                                                        |
|                                     | **Policies**                               |                                                                        |
| [All Bandit Policies][82]           | [Epsilon-Greedy Policy][83]                | [Q-Greedy Policy][84]                                                  |

[75]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1CooperativeModel.html
[76]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1CooperativeMaximumLikelihoodModel.html
[77]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1CooperativeThompsonModel.html

[78]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1FactoredLP.html "Max-norm Projections for Factored MDPs, Guestrin et al."
[79]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1LinearProgramming.html "Multiagent Planning with Factored MDPs, Guestrin et al."
[80]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1JointActionLearner.html "The Dynamics of Reinforcement Learning in Cooperative Multiagent Systems, Claus et al."
[81]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1SparseCooperativeQLearning.html "Sparse Cooperative Q-learning, Kok et al."
[82]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1CooperativePrioritizedSweeping.html "Model-based Multi-Agent Reinforcement Learning with Cooperative Prioritized Sweeping, Bargiacchi et al."

[82]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1BanditPolicyAdaptor.html
[83]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1EpsilonPolicy.html
[84]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1QGreedyPolicy.html

Build Instructions
==================

Dependencies
------------

To build the library you need:

- [cmake](http://www.cmake.org/) >= 3.9
- the [boost library](http://www.boost.org/) >= 1.67
- the [Eigen 3.3 library](http://eigen.tuxfamily.org/index.php?title=Main_Page).
- the [lp\_solve library](http://lpsolve.sourceforge.net/5.5/) (a shared library
  must be available to compile the Python wrapper).

In addition, full C++17 support is now required (**this means at least g++-7**)

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
MAKE_ALL         # Builds all there is to build in the project, but Python.
MAKE_LIB         # Builds the whole core C++ libraries (MDP, POMDP, etc..)
MAKE_MDP         # Builds only the core C++ MDP library
MAKE_FMDP        # Builds only the core C++ Factored/Multi-Agent and MDP libraries
MAKE_POMDP       # Builds only the core C++ POMDP and MDP libraries
MAKE_TESTS       # Builds the library's tests for the compiled core libraries
MAKE_EXAMPLES    # Builds the library's examples using the compiled core libraries
MAKE_PYTHON      # Builds Python bindings for the compiled core libraries
PYTHON_VERSION   # Selects the Python version you want (2 or 3). If not
                 # specified, we try to guess based on your default interpreter.
```

These flags can be combined as needed. For example:

```bash
# Will build MDP and MDP Python 3 bindings
cmake -DCMAKE_BUILD_TYPE=Debug -DMAKE_MDP=1 -DMAKE_PYTHON=1 -DPYTHON_VERSION=3 ..
```

The default flags when nothing is specified are `MAKE_ALL` and
`CMAKE_BUILD_TYPE=Release`.

Note that by default `MAKE_ALL` does not build the Python bindings, as they have
a minor performance hit on the C++ static libraries. You can easily enable them
by using the flag `MAKE_PYTHON`.

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

To compile the library's documentation you need
[Doxygen](http://www.doxygen.nl/). To use it it is sufficient to execute the
following command from the project's root folder:

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
