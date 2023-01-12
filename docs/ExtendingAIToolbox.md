Extending AI Toolbox {#extendingaitoolbox}
====================

[TOC]

Introduction {#intro}
============

This page tries to give a brief description of the concepts behind AIToolbox and
how its code is organized, to help you orient yourself if you want to extend the
library for your specific goals.

Code Organization {#org}
-----------------

Organizing AI algorithms into a consistent interface is not always an easy task,
as the field is extremely diverse, and every algorithm tends to have specific
and unique quirks.

We significantly use the folder structure of the project to group files that
are logically close, and to group algorithms depending on their features. This
hierarchy is somewhat arbitrary, but still better than nothing.

For example, here is (part of) the structure made by the first few levels of
folders/files inside `include/AIToolbox`:

~~~
- Types.hpp
- Utils
- Bandit
    - Policies
    - Types.hpp
- MDP
    - Algorithms
    - Policies
    - Types.hpp
    - Utils.hpp
- POMDP
    - Algorithms
    - Policies
    - Types.hpp
    - Utils.hpp
~~~

Types {#types}
-----

`Types.hpp` files describe the types used in the library, with nested files
describing types that are mostly used in a specific context. AIToolbox tries to
rely standard types (for example `std::vector`) as much as possible, only
resorting to types of external libraries when absolutely necessary.

Utilities {#utils}
---------

`Utils` files and folders contain pure free functions and data-structures that
are (mostly) independent of anything else in the library, aside from types.
These files contain the most common functionality, which is shared between
algorithms and methods, and are grouped together depending on their topic.

For example, in the `include/AIToolbox/Utils` folder one may find utilities for
combinatorics (`Combinatorics.hpp`), probabilities (`Probability.hpp`),
polytopes (`Polytope.hpp`), linear programming (`LP.hpp`), pruning
(`Prune.hpp`), etc. More domain-specific utilities can be found in subfolders
throughout the library.

Free functions try to accept standard types where possible, while otherwise
they accept AIToolbox specific types. Using these functions does not have any
additional code requirements, so they should be easy to use.

What are Algorithms and Policies? {#algopol}
---------------------------------

One of the major distinctions in AIToolbox is between policies and algorithms,
which are stored in separate folders. In simple terms, we denote as policies all
the classes that can directly provide sampling utilities and probability
distributions over actions. Whether they do it by maximizing over a QFunction or
because they are implementing an actor-only method does not matter. We consider
algorithms all other methods.

The reason for this split is that policies have a special place in AIToolbox's
hierarchy, as they are the only classes that are organized using inheritance.
This is because we design policies to be composable: many algorithms that output
policies tend to have other policies as inputs, which makes the idea of
composing arbitrary methods together very attractive. Doing this with templates
becomes burdensome very quickly (as nested type names start growing), so we use
virtual dispatching to be able to pass arbitrary policies around, and to keep
their overall API consistent with virtual interfaces.

On the other hand, in general, algorithms don't have a unified API (although we
do try to keep consistency between similar methods where possible), as they tend
to behave very uniquely.

Mostly Templates {#inherit}
----------------

Aside from policies, we mostly do not introduce inheritance relationships
between classes. Each class is kept as simple and monolithic as possible,
preferring "owns-a" relationships rather than "is-a". Code is sometimes
duplicated between two classes if they are not related (when it can't be made
into a free function), rather than forcing awkward child-parent relationships
for the sake of saving few lines of code. Generalization is achieved using
templates.

This is true for models, algorithms, data structures etc.

Template interfaces are described and checked using template code (which will
hopefully be converted into simpler Concepts as C++20 is released). They mostly
describe the methods that models should have, as the library needs to be able to
feed arbitrary models to algorithms.

--------------------------------------------------------------------------------

Interfaces {#int}
==========

This section lists interfaces that you can implement in your own code to
interoperate with AIToolbox own classes and methods.

Keep in mind that these interfaces must be implemented exactly, down to the
exact type that must be taken/returned; no implicit conversions are allowed. If
these interfaces are not respected, your code will fail to compile.

Experience Interfaces  {#expint}
---------------------

If you are doing RL and want to store experience data in your own class, you can
implement the following methods in your class to use AIToolbox RL models with it
(for example AIToolbox::MDP::MaximumLikelihoodModel).

You do not need to inherit from any class to implement these interfaces.

### Standard Experience ### {#exp}

The basic Experience interface requires you to report the number of seen
transitions, and statistics on the obtained rewards for all state-action pairs.

~~~{.cpp}
    long unsigned getVisits(size_t s,size_t a,size_t s1) const; // Returns the number of seen transitions for the input
    long unsigned getVisitsSum(size_t s,size_t a) const;        // Returns the sum of transition seen from the input state-action pair
    double getReward(size_t s,size_t a) const;                  // Returns the average reward seen for the specified state-action pair
    double getM2(size_t s,size_t a) const;                      // Returns the M2 statistic for the specified state-action pair
~~~

### Eigen Experience Interface ### {#eigexp}

It's possible to return Eigen matrices containing the experience data in order
to speed up the calculations done to update transition and reward functions in
RL models.

~~~{.cpp}
    const M & getVisitsTable(size_t) const; // Returns an SxS Eigen matrix containing the seen transitions using that action
    const M & getRewardMatrix() const;      // Returns an SxA Eigen matrix containing all average rewards
    const M & getM2Matrix() const;          // Returns an SxA Eigen matrix containing all M2 values
~~~

Model Interfaces {#modint}
----------------

In AIToolbox models are glorified data structures. The most complicated thing
they can do is provide an API for sampling, and are otherwise simple wrappers
around transition and reward functions.

Implementing a new model does not require you to inherit from any class in the
library. However, algorithms expect a certain set of methods and parameters to
be accessible.

### Generative MDP Model ### {#genmod}

The most basic basic MDP model interface the generative model, i.e. a model
which we only use to sample new transitions, but where we don't have access to
the underlying transition and reward functions. A simulator would be an
excellent candidate for a generative model.

Here are the methods which you need to implement for your class to be compatible
with algorithms that expect a generative model:

~~~{.cpp}
    size_t getS() const;                                            // Returns the size of the state space
    size_t getA() const;                                            // Returns the size of the action space
    double getDiscount() const;                                     // Returns the discount factor of the MDP
    std::tuple<size_t, double> sampleSR(size_t s,size_t a) const;   // From a state-action pair, return a new state-reward pair.
    bool isTerminal(size_t s) const;                                // Return whether a given state is terminal
~~~

### Standard MDP Model ### {#mod}

Planning algorithms usually require a more strict interface than a generative
model, as they may need access to the underlying transition and reward functions
of your model. In that case, you will need to implement the following
*additional* functions:

~~~{.cpp}
    double getTransitionProbability(size_t s,size_t a,size_t s1) const; // Returns the probability of transitioning from s and a to s1.
    double getExpectedReward(size_t s,size_t a,size_t s1) const;        // Returns the expected reward of the input transition.
~~~

### Eigen MDP Model ### {#eigmod}

An additional set of functions can be implemented in order to unlock
optimizations within the code that leverage the Eigen library. This can
significantly speed up certain algorithms, and may be required for others.

~~~{.cpp}
    const M & getTransitionFunction(size_t a) const; // Returns an SxS Eigen matrix containing the transition probabilities for action a;
    const M & getRewardFunction() const;             // Returns an SxA Eigen matrix containing expected rewards for all state-action pairs.
~~~

Algorithm Interfaces {#algoint}
--------------------

As we mentioned above, algorithms do not generally have a common interface, nor
they inherit from some common ancestor. If you want to implement your own
algorithm, or extend any AIToolbox ones, you have basically complete free rein
on what you want to do.

Policy Interfaces {#polint}
-----------------

In general, if you want to simply implement your own policy using AIToolbox
types, there is nothing special you need to do. You can implement your policy,
use it in your code, and be done.

However, if you need to feed your policy to some algorithm/other policy in the
library, you will have to inherit from one of the interface policy classes, and
implement their virtual methods.

The root of the policy hierarchy is the AIToolbox::PolicyInterface class. This
is a template class, which contains a generic interface that can use arbitrary
types for state and actions (which can be useful for multi-agent environments).

If you are working with MDPs, you can directly inherit its specialized, more
friendly interface: AIToolbox::MDP::PolicyInterface. Once you implement its
virtual functions, you are ready to use the class and don't have to do anything
else.

If your policy takes a QFunction to produce its policy you may want to inherit
from AIToolbox::MDP::QPolicyInterface, which already initializes the QFunction
for you. This is however not required, it's just for convenience.
