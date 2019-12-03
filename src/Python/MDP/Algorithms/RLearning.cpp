#include <AIToolbox/MDP/Algorithms/RLearning.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/MaximumLikelihoodModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseMaximumLikelihoodModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportMDPRLearning() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    class_<RLearning>{"RLearning",

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
                 "Both learning rates must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "@param S The size of the state space.\n"
                 "@param A The size of the action space.\n"
                 "@param alpha The learning rate for the QFunction.\n"
                 "@param rho The learning rate for the average reward."
        , (arg("self"), "S", "A", "alpha", "rho")))

        .def(init<const MaximumLikelihoodModel<Experience>&, optional<double, double>>(
                 "Basic constructor from MaximumLikelihoodModel\n"
                 "\n"
                 "Both learning rates must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not conserve the reference.\n"
                 "\n"
                 "@param model The MDP model that QLearning will use as a base.\n"
                 "@param alpha The learning rate for the QFunction.\n"
                 "@param rho The learning rate for the average reward."
        , (arg("self"), "model", "alpha", "rho")))

        .def(init<const SparseMaximumLikelihoodModel<SparseExperience>&, optional<double, double>>(
                 "Basic constructor from SparseMaximumLikelihoodModel\n"
                 "\n"
                 "Both learning rates must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not conserve the reference.\n"
                 "\n"
                 "@param model The MDP model that QLearning will use as a base.\n"
                 "@param alpha The learning rate for the QFunction.\n"
                 "@param rho The learning rate for the average reward."
        , (arg("self"), "model", "alpha", "rho")))

        .def(init<const Model&, optional<double, double>>(
                 "Basic constructor from Model\n"
                 "\n"
                 "Both learning rates must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not conserve the reference.\n"
                 "\n"
                 "@param model The MDP model that QLearning will use as a base.\n"
                 "@param alpha The learning rate for the QFunction.\n"
                 "@param rho The learning rate for the average reward."
        , (arg("self"), "model", "alpha", "rho")))

        .def(init<const SparseModel&, optional<double, double>>(
                 "Basic constructor from SparseModel\n"
                 "\n"
                 "Both learning rates must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not conserve the reference.\n"
                 "\n"
                 "@param model The MDP model that QLearning will use as a base.\n"
                 "@param alpha The learning rate for the QFunction.\n"
                 "@param rho The learning rate for the average reward."
        , (arg("self"), "model", "alpha", "rho")))

        .def("setAlphaLearningRate",             &RLearning::setAlphaLearningRate,
                 "This function sets the learning rate for the QFunction.\n"
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
                 "@param a The new alpha learning rate parameter."
        , (arg("self"), "a"))

        .def("getAlphaLearningRate",             &RLearning::getAlphaLearningRate,
                 "This function will return the current set alpha learning rate parameter."
        , (arg("self")))

        .def("setRhoLearningRate",             &RLearning::setRhoLearningRate,
                 "This function sets the learning rate parameter for the average reward.\n"
                 "\n"
                 "The learning parameter determines the speed at which the\n"
                 "average reward is modified with respect to new data.\n"
                 "\n"
                 "The learning rate parameter must be > 0.0 and <= 1.0,\n"
                 "otherwise the function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param r The new rho learning rate parameter."
        , (arg("self"), "r"))

        .def("getRhoLearningRate",             &RLearning::getRhoLearningRate,
                 "This function will return the current set rho learning rate parameter."
        , (arg("self")))

        .def("stepUpdateQ",                 &RLearning::stepUpdateQ,
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

        .def("getS",                        &RLearning::getS,
                 "This function returns the number of states on which QLearning is working."
        , (arg("self")))

        .def("getA",                        &RLearning::getA,
                 "This function returns the number of actions on which QLearning is working."
        , (arg("self")))

        .def("getAverageReward",                        &RLearning::getAverageReward,
                 "This function returns the learned average reward."
        , (arg("self")))

        .def("getQFunction",                &RLearning::getQFunction, return_internal_reference<>(),
                 "This function returns a reference to the internal QFunction.\n"
                 "\n"
                 "The returned reference can be used to build Policies, for example\n"
                 "MDP::QGreedyPolicy."
        , (arg("self")));
}

