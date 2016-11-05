#include <AIToolbox/POMDP/Algorithms/POMCP.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

template <typename M>
void exportPOMCPByModel(std::string className) {
    using namespace AIToolbox::POMDP;
    using namespace boost::python;

    using V = POMCP<M>;

    size_t (V::*sampleAction1)(const Belief &, unsigned) = &V::sampleAction;
    size_t (V::*sampleAction2)(size_t, size_t, unsigned) = &V::sampleAction;

    class_<V>{("POMCP" + className).c_str(), (

         "This class represents the POMCP online planner using UCB1 for " + className + ".\n"
         "\n"
         "NOTE: This algorithm is wrapped in Python, but as it uses the internal\n"
         "Models rather than a custom generative model to simulate rollouts it will\n"
         "probably be rather slow for interesting applications. You are of course\n"
         "welcome to try it out, but it is recommended that the generative model\n"
         "is written in C++.\n"
         "\n"
         "This algorithm is an online planner for POMDPs. As an online planner,\n"
         "it needs to have a generative model of the problem. This means that\n"
         "it only needs a way to sample transitions and rewards from the\n"
         "model, but it does not need to know directly the distribution\n"
         "probabilities for them.\n"
         "\n"
         "POMCP plans for a single belief at a time. It follows the logic of\n"
         "Monte Carlo Tree Sampling, where a tree structure is build\n"
         "progressively and action values are deduced as averages of the\n"
         "obtained rewards over rollouts. If the number of sample episodes is\n"
         "high enough, it is guaranteed to converge to the optimal solution.\n"
         "\n"
         "At each rollout, we follow each action and observation within the\n"
         "tree from root to leaves. During this path we chose actions using an\n"
         "algorithm called UCT. What this does is privilege the most promising\n"
         "actions, while guaranteeing that in the limit every action will still\n"
         "be tried an infinite amount of times.\n"
         "\n"
         "Once we arrive to a leaf in the tree, we then expand it with a\n"
         "single new node, representing a new observation we just collected.\n"
         "We then proceed outside the tree following a random policy, but this\n"
         "time we do not track which actions and observations we actually\n"
         "take/obtain. The final reward obtained by this random rollout policy\n"
         "is used to approximate the values for all nodes visited in this\n"
         "rollout inside the tree, before leaving it.\n"
         "\n"
         "Since POMCP expands a tree, it can reuse work it has done if\n"
         "multiple action requests are done in order. To do so, it simply asks\n"
         "for the action that has been performed and its respective obtained\n"
         "observation. Then it simply makes that root branch the new root, and\n"
         "starts again.\n"
         "\n"
         "In order to avoid performing belief updates between each\n"
         "action/observation pair, which can be expensive, POMCP uses particle\n"
         "beliefs. These approximate the beliefs at every step, and are used\n"
         "to select states in the rollouts.\n"
         "\n"
         "A weakness of this implementation is that, as every particle\n"
         "approximation of continuous values, it will lose particles in time.\n"
         "To fight this a possibility is to implement a particle\n"
         "reinvigoration method, which would introduce noise in the particle\n"
         "beliefs in order to keep them 'fresh' (possibly using domain\n"
         "knowledge).").c_str(), no_init}

        .def(init<const M&, size_t, unsigned, double>(
                 "Basic constructor.\n"
                 "\n"
                 "@param m The POMDP model that POMCP will operate upon.\n"
                 "@param beliefSize The size of the initial particle belief.\n"
                 "@param iterations The number of episodes to run before completion.\n"
                 "@param exp The exploration constant. This parameter is VERY important\n"
                 "           to determine the final POMCP performance."
        , (arg("self"), "m", "beliefSize", "iterations", "exp")))

        .def("sampleAction",            sampleAction1,
                 "This function resets the internal graph and samples for the provided state and horizon.\n"
                 "\n"
                 "In general it would be better if the belief did not contain\n"
                 "any terminal states; although not necessary, it would\n"
                 "prevent unnecessary work from being performed.\n"
                 "\n"
                 "@param b The initial belief for the environment.\n"
                 "@param horizon The horizon to plan for.\n"
                 "\n"
                 "@return The best action."
        , (arg("self"), "b", "horizon"))

        .def("sampleAction",            sampleAction2,
                 "This function uses the internal graph to plan.\n"
                 "\n"
                 "This function can be called after a previous call to\n"
                 "sampleAction with a Belief. Otherwise, it will invoke it\n"
                 "anyway with a random belief.\n"
                 "\n"
                 "If a graph is already present though, this function will\n"
                 "select the branch defined by the input action and\n"
                 "observation, and prune the rest. The search will be started\n"
                 "using the existing graph: this should make search faster,\n"
                 "and also not require any belief updates.\n"
                 "\n"
                 "NOTE: Currently there is no particle reinvigoration\n"
                 "implemented, so for long horizons you can expect\n"
                 "progressively degrading performances.\n"
                 "\n"
                 "@param a The action taken in the last timestep.\n"
                 "@param o The observation received in the last timestep.\n"
                 "@param horizon The horizon to plan for.\n"
                 "\n"
                 "@return The best action."
        , (arg("self"), "a", "o", "horizon"))

        .def("setBeliefSize",           &V::setBeliefSize,
                 "This function sets the new size for initial beliefs created from sampleAction().\n"
                 "\n"
                 "Note that this parameter does not bound particle beliefs\n"
                 "created within the tree by result of rollouts: only the ones\n"
                 "directly created from true Beliefs.\n"
                 "\n"
                 "@param beliefSize The new particle belief size."
        , (arg("self"), "beliefSize"))

        .def("setIterations",           &V::setIterations,
                 "This function sets the number of performed rollouts in POMCP."
        , (arg("self"), "iterations"))

        .def("setExploration",          &V::setExploration,
                 "This function sets the new exploration constant for POMCP.\n"
                 "\n"
                 "This parameter is EXTREMELY important to determine POMCP\n"
                 "performance and, ultimately, convergence. In general it is\n"
                 "better to find it empirically, by testing some values and\n"
                 "see which one performs best. Tune this parameter, it really\n"
                 "matters!\n"
                 "\n"
                 "@param exp The new exploration constant."
        , (arg("self"), "exp"))

        .def("getModel",                &V::getModel,   return_value_policy<reference_existing_object>(),
                 "This function returns the POMDP generative model being used."
        , (arg("self")))

        .def("getBeliefSize",           &V::getBeliefSize,
                 "This function returns the initial particle size for converted Beliefs."
        , (arg("self")))

        .def("getIterations",           &V::getIterations,
                 "This function returns the number of iterations performed to plan for an action."
        , (arg("self")))

        .def("getExploration",          &V::getExploration,
                 "This function returns the currently set exploration constant."
        , (arg("self")));
}

void exportPOMDPPOMCP() {
    using namespace AIToolbox::MDP;

    exportPOMCPByModel<POMDPModelBinded>("Model");
    exportPOMCPByModel<POMDPSparseModelBinded>("SparseModel");
}
