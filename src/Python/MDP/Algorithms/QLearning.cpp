#include <AIToolbox/MDP/Algorithms/QLearning.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportMDPQLearning() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    class_<QLearning>{"QLearning",

         "This class represents the QLearning algorithm.\n"
         "\n"
         "This algorithm is a very simple but powerful way to learn the\n"
         "optimal QFunction for an MDP model, where the transition and reward\n"
         "functions are unknown. It works in an offline fashion, meaning that\n"
         "it can be used even if the policy that the agent is currently using\n"
         "is not the optimal one, or is different by the one currently\n"
         "specified by the QLearning QFunction.\n"
         "\n"
         "The idea is to progressively update the QFunction averaging all\n"
         "obtained datapoints. This can be done by generating data via the\n"
         "model, or by simply sending the agent into the world to try stuff\n"
         "out. This allows to avoid modeling directly the transition and\n"
         "reward functions for unknown problems.\n"
         "\n"
         "This algorithm is guaranteed convergence for stationary MDPs (MDPs\n"
         "that do not change their transition and reward functions over time),\n"
         "given that the learning parameter converges to 0 over time.\n"
         "\n"
         "\\sa setLearningRate(double)\n"
         "\n"
         "At the same time, this algorithm can be used for non-stationary\n"
         "MDPs, and it will try to constantly keep up with changes in the\n"
         "environment, given that they are not huge.\n"
         "\n"
         "This algorithm does not actually need to sample from the input\n"
         "model, and so it can be a good algorithm to apply in real world\n"
         "scenarios, where there would be no way to reproduce the world's\n"
         "behavior aside from actually trying out actions. However it is\n"
         "needed to know the size of the state space, the size of the action\n"
         "space and the discount factor of the problem.", no_init}

        .def(init<size_t, size_t, optional<double, double>>(
                 "Basic constructor.\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "@param S The size of the state space.\n"
                 "@param A The size of the action space.\n"
                 "@param discount The discount to use when learning.\n"
                 "@param alpha The learning rate of the QLearning method."
        , (arg("self"), "S", "A", "discount", "alpha")))

        .def(init<const RLModel<Experience>&, optional<double>>(
                 "Basic constructor from RLModel\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not keep the reference, so if the\n"
                 "discount needs to change you'll need to update it here manually\n"
                 "too.\n"
                 "\n"
                 "@param model The MDP model that QLearning will use as a base.\n"
                 "@param alpha The learning rate of the QLearning method."
        , (arg("self"), "model", "alpha")))

        .def(init<const SparseRLModel<SparseExperience>&, optional<double>>(
                 "Basic constructor from SparseRLModel\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters\n"
                 "from the supplied model. It does not conserve the reference.\n"
                 "\n"
                 "@param model The MDP model that QLearning will use as a base.\n"
                 "@param alpha The learning rate of the QLearning method."
        , (arg("self"), "model", "alpha")))

        .def(init<const Model&, optional<double>>(
                 "Basic constructor from Model\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters\n"
                 "from the supplied model. It does not conserve the reference.\n"
                 "\n"
                 "@param model The MDP model that QLearning will use as a base.\n"
                 "@param alpha The learning rate of the QLearning method."
        , (arg("self"), "model", "alpha")))

        .def(init<const SparseModel&, optional<double>>(
                 "Basic constructor from SparseModel\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters\n"
                 "from the supplied model. It does not conserve the reference.\n"
                 "\n"
                 "@param model The MDP model that QLearning will use as a base.\n"
                 "@param alpha The learning rate of the QLearning method."
        , (arg("self"), "model", "alpha")))

        .def("setLearningRate",             &QLearning::setLearningRate,
                 "This function sets the learning rate parameter.\n"
                 "\n"
                 "The learning parameter determines the speed at which the\n"
                 "QFunction is modified with respect to new data. In fully\n"
                 "deterministic environments (such as an agent moving through\n"
                 "a grid, for example), this parameter can be safely set to\n"
                 "1.0 for maximum learning.\n"
                 "\n"
                 "On the other side, in stochastic environments, in order to\n"
                 "converge this parameter should be higher when first starting\n"
                 "to learn, and decrease slowly over time.\n"
                 "\n"
                 "Otherwise it can be kept somewhat high if the environment\n"
                 "dynamics change progressively, and the algorithm will adapt\n"
                 "accordingly. The final behavior of QLearning is very\n"
                 "dependent on this parameter.\n"
                 "\n"
                 "The learning rate parameter must be > 0.0 and <= 1.0,\n"
                 "otherwise the function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param a The new learning rate parameter."
        , (arg("self"), "a"))

        .def("getLearningRate",             &QLearning::getLearningRate,
                 "This function will return the current set learning rate parameter."
        , (arg("self")))

        .def("setDiscount",                 &QLearning::setDiscount,
                 "This function sets the new discount parameter.\n"
                 "\n"
                 "The discount parameter controls the amount that future rewards are considered\n"
                 "by QLearning. If 1, then any reward is the same, if obtained now or in a million\n"
                 "timesteps. Thus the algorithm will optimize overall reward accretion. When less\n"
                 "than 1, rewards obtained in the presents are valued more than future rewards.\n"
                 "\n"
                 "@param d The new discount factor."
        , (arg("self"), "d"))

        .def("getDiscount",                 &QLearning::getDiscount,
                 "This function returns the currently set discount parameter."
        , (arg("self")))

        .def("stepUpdateQ",                 &QLearning::stepUpdateQ,
                 "This function updates the internal QFunction using the discount set during construction.\n"
                 "\n"
                 "This function takes a single experience point and uses it to\n"
                 "update the QFunction. This is a very efficient method to\n"
                 "keep the QFunction up to date with the latest experience.\n"
                 "\n"
                 "@param s The previous state.\n"
                 "@param a The action performed.\n"
                 "@param s1 The new state.\n"
                 "@param rew The reward obtained."
        , (arg("self"), "s", "a", "s1", "rew"))

        .def("getS",                        &QLearning::getS,
                 "This function returns the number of states on which QLearning is working."
        , (arg("self")))

        .def("getA",                        &QLearning::getA,
                 "This function returns the number of actions on which QLearning is working."
        , (arg("self")))

        .def("getQFunction",                &QLearning::getQFunction, return_internal_reference<>(),
                 "This function returns a reference to the internal QFunction.\n"
                 "\n"
                 "The returned reference can be used to build Policies, for example\n"
                 "MDP::QGreedyPolicy."
        , (arg("self")));
}
