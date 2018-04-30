#include <AIToolbox/Factored/MDP/Algorithms/JointActionLearner.hpp>

#include <boost/python.hpp>

void exportFactoredMDPJointActionLearner() {
    using namespace AIToolbox::Factored::MDP;
    using namespace boost::python;
    namespace fm = AIToolbox::Factored;

    class_<JointActionLearner>{"JointActionLearner",

         "This class represents a single Joint Action Learner agent.\n"
         "\n"
         "A JAL agent learns a QFunction for its own values while keeping track of\n"
         "the actions performed by the other agents with which it is interacting.\n"
         "\n"
         "In order to reason about its own QFunction, a JAL keeps a model of the\n"
         "policies of the other agents. This is done by keeping counters for each\n"
         "actions that other agents have performed, and performing a maximum\n"
         "likelihood computation in order to estimate their policies.\n"
         "\n"
         "While internally a QFunction is kept for the full joint action space,\n"
         "after using the policy models the output will be a normal\n"
         "MDP::QFunction, which can then be used to provide a policy.\n"
         "\n"
         "The internal learning is done using MDP::QLearning.\n"
         "\n"
         "This method does not try to handle factorized states. Here we also\n"
         "assume that the joint action space is of reasonable size, as we allocate\n"
         "an MDP::QFunction for it.\n"
         "\n"
         "\\sa AIToolbox::MDP::QLearning", no_init}

        .def(init<size_t, fm::Action, size_t, optional<double, double>>(
                 "Basic constructor.\n"
                 "\n"
                 "@param S The size of the state space.\n"
                 "@param A The size of the joint action space.\n"
                 "@param id The id of this agent in the joint action space.\n"
                 "@param discount The discount factor for the QLearning process.\n"
                 "@param alpha The learning rate for the QLearning process."
        , (arg("self"), "s", "A", "id", "discount", "alpha")))

        .def("stepUpdateQ",             &JointActionLearner::stepUpdateQ,
                 "This function updates the internal joint QFunction.\n"
                 "\n"
                 "This function updates the counts for the actions of the other\n"
                 "agents, and the value of the joint QFunction based on the\n"
                 "inputs.\n"
                 "\n"
                 "Then, it updates the single agent QFunction only for the initial\n"
                 "state using the internal counts to update its expected value\n"
                 "given the new estimates for the other agents' policies.\n"
                 "\n"
                 "@param s The previous state.\n"
                 "@param a The action performed.\n"
                 "@param s1 The new state.\n"
                 "@param rew The reward obtained."
        , (arg("self"), "s", "a", "s1", "rew"))

        .def("getJointQFunction",           &JointActionLearner::getJointQFunction, return_internal_reference<>(),
                 "This function returns the internal joint QFunction.\n"
                 "\n"
                 "@return A reference to the internal joint QFunction."
        , (arg("self")))

        .def("getSingleQFunction",          &JointActionLearner::getSingleQFunction, return_internal_reference<>(),
                 "This function returns the internal single QFunction.\n"
                 "\n"
                 "@return A reference to the internal single QFunction."
        , (arg("self")))

        .def("setLearningRate",             &JointActionLearner::setLearningRate,
                 "This function sets the learning rate parameter.\n"
                 "\n"
                 "The learning parameter determines the speed at which the\n"
                 "QFunction is modified with respect to new data. In fully\n"
                 "deterministic environments (such as an agent moving through\n"
                 "a grid, for example), this parameter can be safely set to\n"
                 "1.0 for maximum learning.\n"
                 "\n"
                 "The learning rate parameter must be > 0.0 and <= 1.0,\n"
                 "otherwise the function will throw an std::invalid_argument.\n"
                 "\n"
                 "\\sa QLearning\n"
                 "\n"
                 "@param a The new learning rate parameter."
        , (arg("self"), "a"))

        .def("getLearningRate",             &JointActionLearner::getLearningRate,
                 "This function will return the current set learning rate parameter.\n"
                 "\n"
                 "@return The currently set learning rate parameter."
        , (arg("self")))

        .def("setDiscount",                 &JointActionLearner::setDiscount,
                 "This function sets the new discount parameter.\n"
                 "\n"
                 "The discount parameter controls the amount that future rewards are considered\n"
                 "by QLearning. If 1, then any reward is the same, if obtained now or in a million\n"
                 "timesteps. Thus the algorithm will optimize overall reward accretion. When less\n"
                 "than 1, rewards obtained in the presents are valued more than future rewards.\n"
                 "\n"
                 "\\sa QLearning\n"
                 "\n"
                 "@param d The new discount factor."
        , (arg("self"), "d"))

        .def("getDiscount",                 &JointActionLearner::getDiscount,
                 "This function returns the currently set discount parameter.\n"
                 "\n"
                 "@return The currently set discount parameter."
        , (arg("self")))

        .def("getS",                        &JointActionLearner::getS,
                 "This function returns the number of states on which JointActionLearner is working.\n"
                 "\n"
                 "@return The number of states."
        , (arg("self")))

        .def("getA",                        &JointActionLearner::getA, return_internal_reference<>(),
                 "This function returns the action space on which JointActionLearner is working.\n"
                 "\n"
                 "@return The action space."
        , (arg("self")))

        .def("getId",                        &JointActionLearner::getId,
                 "This function returns the id of the agent represented by this class.\n"
                 "\n"
                 "@return The id of this agent."
        , (arg("self")));
}

