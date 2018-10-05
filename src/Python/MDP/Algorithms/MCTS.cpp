#include <AIToolbox/MDP/Algorithms/MCTS.hpp>

#include <AIToolbox/MDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include "../GenerativeModelPython.hpp"

#include <boost/python.hpp>

template <typename M>
void exportMCTSByModel(std::string className) {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    using V = MCTS<M>;

    size_t (V::*sampleAction1)(size_t, unsigned) = &V::sampleAction;
    size_t (V::*sampleAction2)(size_t, size_t, unsigned) = &V::sampleAction;

    class_<V>{("MCTS" + className).c_str(), (

         "This class represents the MCTS online planner using UCB1 for " + className + ".\n"
         "\n"
         "NOTE: This algorithm is wrapped in Python, but as it uses the internal\n"
         "Models rather than a custom generative model to simulate rollouts it will\n"
         "probably be rather slow for interesting applications. You are of course\n"
         "welcome to try it out, but it is recommended that the generative model\n"
         "is written in C++.\n"
         "\n"
         "This algorithm is an online planner for MDPs. As an online planner,\n"
         "it needs to have a generative model of the problem. This means that\n"
         "it only needs a way to sample transitions and rewards from the\n"
         "model, but it does not need to know directly the distribution\n"
         "probabilities for them.\n"
         "\n"
         "MCTS plans for a single state at a time. It builds a tree structure\n"
         "progressively and action values are deduced as averages of the\n"
         "obtained rewards over rollouts. If the number of sample episodes is\n"
         "high enough, it is guaranteed to converge to the optimal solution.\n"
         "\n"
         "At each rollout, we follow each action and resulting state within the\n"
         "tree from root to leaves. During this path we chose actions using an\n"
         "algorithm called UCT. What this does is privilege the most promising\n"
         "actions, while guaranteeing that in the limit every action will still\n"
         "be tried an infinite amount of times.\n"
         "\n"
         "Once we arrive to a leaf in the tree, we then expand it with a\n"
         "single new node, representing a new state for the path we just\n"
         "followed. We then proceed outside the tree following a random\n"
         "policy, but this time we do not track which actions and states\n"
         "we actually experience. The final reward obtained by this random\n"
         "rollout policy is used to approximate the values for all nodes\n"
         "visited in this rollout inside the tree, before leaving it.\n"
         "\n"
         "Since MCTS expands a tree, it can reuse work it has done if\n"
         "multiple action requests are done in order. To do so, it simply asks\n"
         "for the action that has been performed and its respective new state.\n"
         "Then it simply makes that root branch the new root, and starts\n"
         "again." ).c_str(), no_init}

        .def(init<const M&, unsigned, double>(
                 "Basic constructor.\n"
                 "\n"
                 "@param m The MDP model that MCTS will operate upon.\n"
                 "@param iterations The number of episodes to run before completion.\n"
                 "@param exp The exploration constant. This parameter is VERY important\n"
                 "           to determine the final MCTS performance."
        , (arg("self"), "m", "iterations", "exp")))

        .def("sampleAction",            sampleAction1,
                 "This function resets the internal graph and samples for the provided state and horizon.\n"
                 "\n"
                 "@param s The initial state for the environment.\n"
                 "@param horizon The horizon to plan for.\n"
                 "\n"
                 "@return The best action."
        , (arg("self"), "s", "horizon"))

        .def("sampleAction",            sampleAction2,
                 "This function uses the internal graph to plan.\n"
                 "\n"
                 "This function can be called after a previous call to\n"
                 "sampleAction with a state. Otherwise, it will invoke it\n"
                 "anyway with the provided next state.\n"
                 "\n"
                 "If a graph is already present though, this function will\n"
                 "select the branch defined by the input action and\n"
                 "observation, and prune the rest. The search will be started\n"
                 "using the existing graph: this should make search faster.\n"
                 "\n"
                 "@param a The action taken in the last timestep.\n"
                 "@param s1 The state experienced after the action was taken.\n"
                 "@param horizon The horizon to plan for.\n"
                 "\n"
                 "@return The best action."
        , (arg("self"), "a", "s1", "horizon"))

        .def("setIterations",           &V::setIterations,
                 "This function sets the number of performed rollouts in MCTS."
        , (arg("self"), "iterations"))

        .def("setExploration",          &V::setExploration,
                 "This function sets the new exploration constant for MCTS.\n"
                 "\n"
                 "This parameter is EXTREMELY important to determine MCTS\n"
                 "performance and, ultimately, convergence. In general it is\n"
                 "better to find it empirically, by testing some values and\n"
                 "see which one performs best. Tune this parameter, it really\n"
                 "matters!\n"
                 "\n"
                 "@param exp The new exploration constant."
        , (arg("self"), "exp"))

        .def("getModel",                &V::getModel,   return_value_policy<reference_existing_object>(),
                 "This function returns the MDP generative model being used."
        , (arg("self")))

        .def("getIterations",           &V::getIterations,
                 "This function returns the number of iterations performed to plan for an action."
        , (arg("self")))

        .def("getExploration",          &V::getExploration,
                 "This function returns the currently set exploration constant."
        , (arg("self")));
}

void exportMDPMCTS() {
    using namespace AIToolbox::MDP;

    exportMCTSByModel<RLModel<Experience>>("RLModel");
    exportMCTSByModel<SparseRLModel<SparseExperience>>("SparseRLModel");
    exportMCTSByModel<Model>("Model");
    exportMCTSByModel<SparseModel>("SparseModel");
    exportMCTSByModel<GenerativeModelPython>("GenerativeModelPython");
}
