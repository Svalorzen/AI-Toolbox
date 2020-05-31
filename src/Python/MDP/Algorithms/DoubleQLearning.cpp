#include <AIToolbox/MDP/Algorithms/DoubleQLearning.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/MaximumLikelihoodModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseMaximumLikelihoodModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportMDPDoubleQLearning() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    class_<DoubleQLearning>{"DoubleQLearning",

         "This class represents the double QLearning algorithm.\n"
         "\n"
         "The QLearning algorithm is biased to overestimate the expected future\n"
         "reward during the Bellman equation update, as the bootstrapped max over\n"
         "the same QFunction is actually an unbiased estimator for the expected\n"
         "max, rather than the max expected.\n"
         "\n"
         "This is a problem for certain classes of problems, and DoubleQLearning\n"
         "tries to fix that.\n"
         "\n"
         "DoubleQLearning maintains two separate QFunctions, and in a given\n"
         "timestep one is selected randomly to be updated. The update has the same\n"
         "form as the standard QLearning update, except that the *other* QFunction\n"
         "is used to estimate the expected future reward. The math shows that this\n"
         "technique still results in a bias estimation, but in this case we tend\n"
         "to underestimate.\n"
         "\n"
         "We can still try to counteract this with optimistic initialization, and\n"
         "the final result is often more stable than simple QLearning.\n"
         "\n"
         "Since action selection should be performed w.r.t. both QFunctions,\n"
         "DoubleQLearning stores two things: the first QFunction, and the sum\n"
         "between the first QFunction and the second. The second QFunction is not\n"
         "stored explicitly, and is instead always computed on-the-fly when\n"
         "needed.\n"
         "\n"
         "We do this so we can easily return the sum of both QFunction to apply a\n"
         "Policy to, without the need to store three separate QFunctions\n"
         "explicitly (lowering a bit the memory requirements).\n"
         "\n"
         "If you are interested in the actual values stored in the two 'main'\n"
         "QFunctions, please use getQFunctionA() and getQFunctionB(). Note that\n"
         "getQFunctionB() will not return a reference!", no_init}

        .def(init<size_t, size_t, optional<double, double>>(
                 "Basic constructor.\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "@param S The size of the state space.\n"
                 "@param A The size of the action space.\n"
                 "@param discount The discount to use when learning.\n"
                 "@param alpha The learning rate of the DoubleQLearning method."
        , (arg("self"), "S", "A", "discount", "alpha")))

        .def(init<const MaximumLikelihoodModel<Experience>&, optional<double>>(
                 "Basic constructor from MaximumLikelihoodModel\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not keep the reference, so if the\n"
                 "discount needs to change you'll need to update it here manually\n"
                 "too.\n"
                 "\n"
                 "@param model The MDP model that DoubleQLearning will use as a base.\n"
                 "@param alpha The learning rate of the DoubleQLearning method."
        , (arg("self"), "model", "alpha")))

        .def(init<const SparseMaximumLikelihoodModel<SparseExperience>&, optional<double>>(
                 "Basic constructor from SparseMaximumLikelihoodModel\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters\n"
                 "from the supplied model. It does not conserve the reference.\n"
                 "\n"
                 "@param model The MDP model that DoubleQLearning will use as a base.\n"
                 "@param alpha The learning rate of the DoubleQLearning method."
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
                 "@param model The MDP model that DoubleQLearning will use as a base.\n"
                 "@param alpha The learning rate of the DoubleQLearning method."
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
                 "@param model The MDP model that DoubleQLearning will use as a base.\n"
                 "@param alpha The learning rate of the DoubleQLearning method."
        , (arg("self"), "model", "alpha")))

        .def("setLearningRate",             &DoubleQLearning::setLearningRate,
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
                 "accordingly. The final behavior of DoubleQLearning is very\n"
                 "dependent on this parameter.\n"
                 "\n"
                 "The learning rate parameter must be > 0.0 and <= 1.0,\n"
                 "otherwise the function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param a The new learning rate parameter."
        , (arg("self"), "a"))

        .def("getLearningRate",             &DoubleQLearning::getLearningRate,
                 "This function will return the current set learning rate parameter."
        , (arg("self")))

        .def("setDiscount",                 &DoubleQLearning::setDiscount,
                 "This function sets the new discount parameter.\n"
                 "\n"
                 "The discount parameter controls the amount that future rewards are considered\n"
                 "by DoubleQLearning. If 1, then any reward is the same, if obtained now or in a million\n"
                 "timesteps. Thus the algorithm will optimize overall reward accretion. When less\n"
                 "than 1, rewards obtained in the presents are valued more than future rewards.\n"
                 "\n"
                 "@param d The new discount factor."
        , (arg("self"), "d"))

        .def("getDiscount",                 &DoubleQLearning::getDiscount,
                 "This function returns the currently set discount parameter."
        , (arg("self")))

        .def("stepUpdateQ",                 &DoubleQLearning::stepUpdateQ,
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

        .def("getS",                        &DoubleQLearning::getS,
                 "This function returns the number of states on which DoubleQLearning is working."
        , (arg("self")))

        .def("getA",                        &DoubleQLearning::getA,
                 "This function returns the number of actions on which DoubleQLearning is working."
        , (arg("self")))

        .def("getQFunction",                &DoubleQLearning::getQFunction, return_internal_reference<>(),
                 "This function returns a reference to the internal 'sum' QFunction.\n"
                 "\n"
                 "The QFunction that is returned does not contain 'true' values,\n"
                 "but instead is the sum of the two QFunctions that are being\n"
                 "updated by DoubleQLearning. This is to make it possible to\n"
                 "select actions using standard policy classes.\n"
                 "\n"
                 "The returned reference can be used to build Policies, for example\n"
                 "MDP::QGreedyPolicy.\n"
                 "\n"
                 "@return The internal 'sum' QFunction."
        , (arg("self")))

        .def("getQFunctionA",                &DoubleQLearning::getQFunctionA, return_internal_reference<>(),
                 "This function returns a reference to the first internal QFunction.\n"
                 "\n"
                 "The returned reference can be used to build Policies, for\n"
                 "example MDP::QGreedyPolicy, but you should probably use\n"
                 "getQFunction() for that.\n"
                 "\n"
                 "@return The internal first QFunction."
        , (arg("self")))

        .def("getQFunctionB",                &DoubleQLearning::getQFunctionB,
                 "This function returns a copy to the second QFunction.\n"
                 "\n"
                 "This QFunction is constructed on the fly, and so is not returned by reference!\n"
                 "\n"
                 "@return What the second QFunction should be."
        , (arg("self")));
}

