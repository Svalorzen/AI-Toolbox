AI-Toolbox
==========

[![AI-Toolbox](https://github.com/Svalorzen/AI-Toolbox/actions/workflows/build_cmake.yml/badge.svg)](https://github.com/Svalorzen/AI-Toolbox/actions/workflows/build_cmake.yml)

[![Library overview video](https://user-images.githubusercontent.com/1609228/99919181-3404dc00-2d1c-11eb-8593-0bf6af44cef8.png)](https://www.youtube.com/watch?v=qjSo41DVSXg)

This C++ toolbox is aimed at representing and solving common AI problems,
implementing an easy-to-use interface which should be hopefully extensible
to many problems, while keeping code readable.

Current development includes MDPs, POMDPs and related algorithms. This toolbox
was originally developed taking inspiration from the Matlab `MDPToolbox`, which
you can find [here](https://miat.inrae.fr/MDPtoolbox/), and from the
`pomdp-solve` software written by A. R. Cassandra, which you can find
[here](http://www.pomdp.org/code/index.shtml).

If you are new to the field of reinforcement learning, we have a few [simple
tutorials](http://svalorzen.github.io/AI-Toolbox/tutorials.html) that can help
you get started. An excellent, more in depth introduction to the basics of
reinforcement learning can be found freely online in [this
book](http://incompleteideas.net/book/ebook/the-book.html).

If you use this toolbox for research, please consider citing our [JMLR
article](https://www.jmlr.org/papers/volume21/18-402/18-402.pdf):

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

Example
=======

```cpp
// The model can be any custom class that respects a 10-method interface.
auto model = makeTigerProblem();
unsigned horizon = 10; // The horizon of the solution.

// The 0.0 is the convergence parameter. It gives a way to stop the
// computation if the policy has converged before the horizon.
AIToolbox::POMDP::IncrementalPruning solver(horizon, 0.0);

// Solve the model and obtain the optimal value function.
auto [bound, valueFunction] = solver(model);

// We create a policy from the solution to compute the agent's actions.
// The parameters are the size of the model (SxAxO), and the value function.
AIToolbox::POMDP::Policy policy(2, 3, 2, valueFunction);

// We begin a simulation with a uniform belief. We sample from the belief
// in order to get a "real" state for the world, since this code has to
// both emulate the environment and control the agent.
AIToolbox::POMDP::Belief b(2); b << 0.5, 0.5;
auto s = AIToolbox::sampleProbability(b.size(), b, rand);

// We sample the first action. The id is to follow the policy tree later.
auto [a, id] = policy.sampleAction(b, horizon);

double totalReward = 0.0;// As an example, we store the overall reward.
for (int t = horizon - 1; t >= 0; --t) {
    // We advance the world one step.
    auto [s1, o, r] = model.sampleSOR(s, a);
    totalReward += r;

    // We select our next action from the observation we got.
    std::tie(a, id) = policy.sampleAction(id, o, t);

    s = s1; // Finally we update the world for the next timestep.
}
```

Documentation
=============

The latest documentation is available [here](http://svalorzen.github.io/AI-Toolbox/).
We have a few [tutorials](http://svalorzen.github.io/AI-Toolbox/tutorials.html)
that can help you get started with the toolbox. The tutorials are in C++, but
the `examples` folder contains equivalent Python code which you can follow
along just as well.

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
this library. [Here's the docs on that](http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1CassandraParser.html).

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
enumerate here. In particular, we have utilities for [combinatorics][comb],
[polytopes][poly], [linear programming][lipo], [sampling and distributions][dist],
[automated statistics][stat], [belief updating][belu], [many][trie] [data][fgra] [structures][fmat],
[logging][logg], [seeding][seed] and much more.

[comb]: http://svalorzen.github.io/AI-Toolbox/Combinatorics_8hpp.html
[poly]: http://svalorzen.github.io/AI-Toolbox/Polytope_8hpp.html
[lipo]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1LP.html
[dist]: http://svalorzen.github.io/AI-Toolbox/Probability_8hpp.html
[stat]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Statistics.html
[belu]: http://svalorzen.github.io/AI-Toolbox/include_2AIToolbox_2POMDP_2Utils_8hpp.html
[trie]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Trie.html
[fgra]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1FactorGraph.html
[fmat]: http://svalorzen.github.io/AI-Toolbox/FactoredMatrix_8hpp.html
[logg]: http://svalorzen.github.io/AI-Toolbox/logging.html
[seed]: http://svalorzen.github.io/AI-Toolbox/Seeder_8hpp.html

### Bandit/Normal Games: ###

|                                                            | **Models**                                         |                                     |
| :--------------------------------------------------------: | :------------------------------------------------: | :---------------------------------: |
| [Basic Model][bmod]                                        |                                                    |                                     |
|                                                            | **Policies**                                       |                                     |
| [Exploring Selfish Reinforcement Learning (ESRL)][esrl]    | [Q-Greedy Policy][bqgr]                            | [Softmax Policy][bsof]              |
| [Linear Reward Penalty][lrpe]                              | [Thompson Sampling (Student-t distribution)][btho] | [Random Policy][brnd]               |
| [Top-Two Thompson Sampling (Student-t distribution)][ttho] | [Successive Rejects][sure]                         | [T3C (Normal distribution)][t3cp]   |

[bmod]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1Model.html "Reinforcement Learning: An Introduction, Ch 2.1, Sutton & Barto"
[esrl]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1ESRLPolicy.html "Exploring selfish reinforcement learning in repeated games with stochastic rewards, Verbeeck et al."
[bqgr]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1QGreedyPolicy.html "A Tutorial on Thompson Sampling, Russo et al."
[bsof]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1QSoftmaxPolicy.html "Reinforcement Learning: An Introduction, Ch 2.3, Sutton & Barto"
[lrpe]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1LRPPolicy.html "Self-organization in large populations of mobile robots, Ch 3: Stochastic Learning Automata, Unsal"
[btho]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1ThompsonSamplingPolicy.html "Thompson Sampling for 1-Dimensional Exponential Family Bandits, Korda et al."
[brnd]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1RandomPolicy.html
[ttho]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1TopTwoThompsonSamplingPolicy.html "Simple Bayesian Algorithms for Best Arm Identification, Russo"
[sure]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1SuccessiveRejectsPolicy.html "Best Arm Identification in Multi-Armed Bandits, Audibert et al."
[t3cp]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Bandit_1_1T3CPolicy.html "Fixed-confidence guarantees for Bayesian best-arm identification, Shang et al."

### Single Agent MDP/Stochastic Games: ###

|                                         | **Models**                                                   |                                                   |
| :-------------------------------------: | :----------------------------------------------------------: | :-----------------------------------------------: |
| [Basic Model][mmod]                     | [Sparse Model][msmo]                                         | [Maximum Likelihood Model][mmlm]                  |
| [Sparse Maximum Likelihood Model][msml] | [Thompson Model (Dirichlet + Student-t distributions)][mtmo] |                                                   |
|                                         | **Algorithms**                                               |                                                   |
| [Dyna-Q][dynq]                          | [Dyna2][dyn2]                                                | [Expected SARSA][esar]                            |
| [Hysteretic Q-Learning][hqle]           | [Importance Sampling][imsa]                                  | [Linear Programming][m-lp]                        |
| [Monte Carlo Tree Search (MCTS)][mcts]  | [Policy Evaluation][mpoe]                                    | [Policy Iteration][mpoi]                          |
| [Prioritized Sweeping][mprs]            | [Q-Learning][qlea]                                           | [Double Q-Learning][dqle]                         |
| [Q(λ)][qlam]                            | [R-Learning][rlea]                                           | [SARSA(λ)][sarl]                                  |
| [SARSA][sars]                           | [Retrace(λ)][retl]                                           | [Tree Backup(λ)][trel]                            |
| [Value Iteration][vait]                 |                                                              |                                                   |
|                                         | **Policies**                                                 |                                                   |
| [Basic Policy][mpol]                    | [Epsilon-Greedy Policy][megr]                                | [Softmax Policy][msof]                            |
| [Q-Greedy Policy][mqgr]                 | [PGA-APP][pgaa]                                              | [Win or Learn Fast Policy Iteration (WoLF)][wolf] |

[mmod]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1Model.html
[msmo]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1SparseModel.html
[mmlm]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1MaximumLikelihoodModel.html
[msml]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1SparseMaximumLikelihoodModel.html
[mtmo]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1ThompsonModel.html

[dynq]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1DynaQ.html "Reinforcement Learning: An Introduction, Ch 9.2, Sutton & Barto"
[dyn2]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1Dyna2.html "Sample-Based Learning and Search with Permanent and Transient Memories, Silver et al."
[esar]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1ExpectedSARSA.html "A Theoretical and Empirical Analysis of Expected Sarsa, van Seijen et al."
[hqle]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1HystereticQLearning.html "Hysteretic Q-Learning : an algorithm for decentralized reinforcement learning in cooperative multi-agent teams, Matignon et al."
[imsa]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1ImportanceSampling.html "Eligibility Traces for Off-Policy Policy Evaluation, Precup"
[m-lp]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1LinearProgramming.html
[mcts]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1MCTS.html "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search, Coulom"
[mpoe]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1PolicyEvaluation.html "Reinforcement Learning: An Introduction, Ch 4.1, Sutton & Barto"
[mpoi]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1PolicyIteration.html "Reinforcement Learning: An Introduction, Ch 4.3, Sutton & Barto"
[mprs]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1PrioritizedSweeping.html "Reinforcement Learning: An Introduction, Ch 9.4, Sutton & Barto"
[qlea]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1QLearning.html "Reinforcement Learning: An Introduction, Ch 6.5, Sutton & Barto"
[dqle]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1DoubleQLearning.html "Double Q-learning, van Hasselt"
[qlam]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1QL.html "Q(λ) with Off-Policy Corrections, Harutyunyan et al."
[rlea]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1RLearning.html "A Reinforcement Learning Method for Maximizing Undiscounted Rewards, Schwartz"
[sarl]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1SARSAL.html "Reinforcement Learning: An Introduction, Ch 7.5, Sutton & Barto"
[sars]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1SARSA.html "Reinforcement Learning: An Introduction, Ch 6.4, Sutton & Barto"
[retl]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1RetraceL.html "Safe and efficient off-policy reinforcement learning, Munos et al."
[trel]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1TreeBackupL.html "Eligibility Traces for Off-Policy Policy Evaluation, Precup"
[vait]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1ValueIteration.html "Reinforcement Learning: An Introduction, Ch 4.4, Sutton & Barto"

[mpol]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1Policy.html
[megr]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1EpsilonPolicy.html
[msof]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1QSoftmaxPolicy.html
[mqgr]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1QGreedyPolicy.html
[pgaa]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1PGAAPPPolicy.html "Multi-Agent Learning with Policy Prediction, Zhang et al."
[wolf]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1MDP_1_1WoLFPolicy.html "Rational and Convergent Learning in Stochastic Games, Bowling et al."

### Single Agent POMDP: ###

|                              | **Models**                                    |                                            |
| :--------------------------: | :-------------------------------------------: | :----------------------------------------: |
| [Basic Model][pmod]          | [Sparse Model][pmsm]                          |                                            |
|                              | **Algorithms**                                |                                            |
| [Augmented MDP (AMDP)][amdp] | [Blind Strategies][blin]                      | [Fast Informed Bound][faib]                |
| [GapMin][gapm]               | [Incremental Pruning][incp]                   | [Linear Support][lisu]                     |
| [PERSEUS][pers]              | [POMCP with UCB1][pomc]                       | [Point Based Value Iteration (PBVI)][pbvi] |
| [QMDP][qmdp]                 | [Real-Time Belief State Search (RTBSS)][rtbs] | [SARSOP][ssop]                             |
| [Witness][witn]              | [rPOMCP][rpom]                                |                                            |
|                              | **Policies**                                  |                                            |
| [Basic Policy][ppol]         |                                               |                                            |

[pmod]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1Model.html
[pmsm]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1SparseModel.html

[amdp]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1AMDP.html "Probabilistic robotics, Ch 16: Approximate POMDP Techniques, Thrun"
[blin]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1BlindStrategies.html "Incremental methods for computing bounds in partially observable Markov decision processes, Hauskrecht"
[faib]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1FastInformedBound.html "Value-Function Approximations for Partially Observable Markov Decision Processes, Hauskrecht"
[gapm]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1GapMin.html "Closing the Gap: Improved Bounds on Optimal POMDP Solutions, Poupart et al."
[incp]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1IncrementalPruning.html "Incremental Pruning: A Simple, Fast, Exact Method for Partially Observable Markov Decision Processes, Cassandra et al."
[lisu]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1LinearSupport.html "Algorithms for Partially Observable Markov Decision Processes, Phd Thesis, Cheng"
[pers]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1PERSEUS.html "Perseus: Randomized Point-based Value Iteration for POMDPs, Spaan et al."
[pomc]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1POMCP.html "Monte-Carlo Planning in Large POMDPs, Silver et al."
[pbvi]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1PBVI.html "Point-based value iteration: An anytime algorithm for POMDPs, Pineau et al."
[qmdp]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1QMDP.html "Probabilistic robotics, Ch 16: Approximate POMDP Techniques, Thrun"
[rtbs]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1RTBSS.html "Real-Time Decision Making for Large POMDPs, Paquet et al."
[ssop]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1SARSOP.html "SARSOP: Efficient Point-Based POMDP Planning by Approximating Optimally Reachable Belief Spaces, Kurniawati et al."
[witn]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1Witness.html "Planning and acting in partially observable stochastic domains, Kaelbling et al."
[rpom]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1rPOMCP.html "Dynamic Resource Allocation for Multi-Camera Systems, Phd Thesis, Bargiacchi"

[ppol]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1POMDP_1_1Policy.html

### Factored/Joint Multi-Agent: ###

#### Bandits: ####

Not in Python yet.

|                                                          | **Models**                                                     |                                                      |
| :------------------------------------------------------: | :------------------------------------------------------------: | :--------------------------------------------------: |
| [Basic Model][fbmo]                                      | [Flattened Model][fbfm]                                        |                                                      |
|                                                          | **Algorithms**                                                 |                                                      |
| [Max-Plus][mplu]                                         | [Multi-Objective Variable Elimination (MOVE)][move]            | [Upper Confidence Variable Elimination (UCVE)][ucve] |
| [Variable Elimination][vael]                             | [Local Search][lose]                                           | [Reusing Iterative Local Search][rils]               |
|                                                          | **Policies**                                                   |                                                      |
| [Q-Greedy Policy][fbqg]                                  | [Random Policy][fbra]                                          | [Learning with Linear Rewards (LLR)][llre]           |
| [Multi-Agent Upper Confidence Exploration (MAUCE)][mauc] | [Multi-Agent Thompson-Sampling (Student-t distribution)][mats] | [Multi-Agent RMax (MARMax)][mmax]                    |
| [Single-Action Policy][fbsa]                             |

[fbmo]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1Model.html "Learning to Coordinate with Coordination Graphs in Repeated Single-Stage Multi-Agent Decision Problems, Bargiacchi et al."
[fbfm]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1FlattenedModel.html "Learning to Coordinate with Coordination Graphs in Repeated Single-Stage Multi-Agent Decision Problems, Bargiacchi et al."

[mplu]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1MaxPlus.html "Collaborative Multiagent Reinforcement Learning by Payoff Propagation, Kok et al."
[move]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1MultiObjectiveVariableElimination.html "Multi-Objective Variable Elimination for Collaborative Graphical Games, Roijers et al."
[ucve]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1UCVE.html "Learning to Coordinate with Coordination Graphs in Repeated Single-Stage Multi-Agent Decision Problems, Bargiacchi et al."
[vael]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1VariableElimination.html "Multiagent Planning with Factored MDPs, Guestrin et al."
[lose]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1LocalSearch.html "Heuristic Coordination in Cooperative Multi-Agent Reinforcement Learning, Petri et al."
[rils]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1ReusingIterativeLocalSearch.html "Heuristic Coordination in Cooperative Multi-Agent Reinforcement Learning, Petri et al."

[fbqg]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1QGreedyPolicy.html
[fbra]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1RandomPolicy.html
[llre]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1LLRPolicy.html "Combinatorial Network Optimization with Unknown Variables: Multi-Armed Bandits with Linear Rewards, Gai et al."
[mauc]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1MAUCEPolicy.html "Learning to Coordinate with Coordination Graphs in Repeated Single-Stage Multi-Agent Decision Problems, Bargiacchi et al."
[mats]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1ThompsonSamplingPolicy.html "Multi-Agent Thompson Sampling for Bandit Applications with Sparse Neighbourhood Structures, Verstraeten et al."
[mmax]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1MARMaxPolicy.html "Multi-agent RMax for Multi-Agent Multi-Armed Bandits, Bargiacchi et al."
[fbsa]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1Bandit_1_1SingleActionPolicy.html

#### MDP: ####

Not in Python yet.

|                                       | **Models**                                   |                                                                          |
| :-----------------------------------: | :------------------------------------------: | :----------------------------------------------------------------------: |
| [Cooperative Basic Model][fmcm]       | [Cooperative Maximum Likelihood Model][fmml] | [Cooperative Thompson Model (Dirichlet + Student-t distributions)][fmtm] |
|                                       | **Algorithms**                               |                                                                          |
| [FactoredLP][falp]                    | [Multi Agent Linear Programming][malp]       | [Joint Action Learners][jale]                                            |
| [Sparse Cooperative Q-Learning][scql] | [Cooperative Prioritized Sweeping][cops]     |                                                                          |
|                                       | **Policies**                                 |                                                                          |
| [All Bandit Policies][fmbp]           | [Epsilon-Greedy Policy][fmeg]                | [Q-Greedy Policy][fmqg]                                                  |

[fmcm]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1CooperativeModel.html
[fmml]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1CooperativeMaximumLikelihoodModel.html
[fmtm]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1CooperativeThompsonModel.html

[falp]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1FactoredLP.html "Max-norm Projections for Factored MDPs, Guestrin et al."
[malp]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1LinearProgramming.html "Multiagent Planning with Factored MDPs, Guestrin et al."
[jale]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1JointActionLearner.html "The Dynamics of Reinforcement Learning in Cooperative Multiagent Systems, Claus et al."
[scql]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1SparseCooperativeQLearning.html "Sparse Cooperative Q-learning, Kok et al."
[cops]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1CooperativePrioritizedSweeping.html "Model-based Multi-Agent Reinforcement Learning with Cooperative Prioritized Sweeping, Bargiacchi et al."

[fmbp]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1BanditPolicyAdaptor.html
[fmeg]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1EpsilonPolicy.html
[fmqg]: http://svalorzen.github.io/AI-Toolbox/classAIToolbox_1_1Factored_1_1MDP_1_1QGreedyPolicy.html

Build Instructions
==================

Dependencies
------------

To build the library you need:

- [cmake](http://www.cmake.org/) >= 3.12
- the [boost library](http://www.boost.org/) >= 1.67
- the [Eigen 3.3 library](http://eigen.tuxfamily.org/index.php?title=Main_Page).
- the [lp\_solve library](http://lpsolve.sourceforge.net/5.5/) (a shared library
  must be available to compile the Python wrapper).

In addition, C++20 support is now required (**this means at least g++-10**)

On a Ubuntu system, you can install these dependencies with the following
command:

```bash
sudo apt install g++-10 cmake libboost1.71-all-dev liblpsolve55-dev lp-solve libeigen3-dev
```

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
CMAKE_BUILD_TYPE   # Defines the build type
MAKE_ALL           # Builds all there is to build in the project, but Python.
MAKE_LIB           # Builds the whole core C++ libraries (MDP, POMDP, etc..)
MAKE_MDP           # Builds only the core C++ MDP library
MAKE_FMDP          # Builds only the core C++ Factored/Multi-Agent and MDP libraries
MAKE_POMDP         # Builds only the core C++ POMDP and MDP libraries
MAKE_TESTS         # Builds the library's tests for the compiled core libraries
MAKE_EXAMPLES      # Builds the library's examples using the compiled core libraries
MAKE_PYTHON        # Builds Python bindings for the compiled core libraries
AI_PYTHON_VERSION  # Selects the Python version you want (2 or 3). If not
                   #   specified, we try to guess based on your default interpreter.
AI_LOGGING_ENABLED # Whether the library logging code is enabled at runtime.
```

These flags can be combined as needed. For example:

```bash
# Will build MDP and MDP Python 3 bindings
cmake -DCMAKE_BUILD_TYPE=Debug -DMAKE_MDP=1 -DMAKE_PYTHON=1 -DAI_PYTHON_VERSION=3 ..
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

For an extensive pre-made setup of a C++/CMake project using AI-Toolbox *on
Linux*, please do checkout [this
repository](https://github.com/Svalorzen/AI-Toolbox-Experiments). It contains
the setup I personally use when working with AI-Toolbox. It also comes with many
additional tools you might need, which are nevertheless all optional.

Alternatively, to compile a program that uses this library, simply link it
against the compiled libraries you need, and possibly to the `lp_solve`
libraries (if using POMDP or FMDP).

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
