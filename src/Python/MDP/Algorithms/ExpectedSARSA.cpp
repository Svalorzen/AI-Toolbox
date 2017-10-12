#include <AIToolbox/MDP/Algorithms/ExpectedSARSA.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportMDPExpectedSARSA() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    class_<ExpectedSARSA>{"ExpectedSARSA",

         "This algorithm is a subtle improvement over the SARSA algorithm.\n"
         "\n"
         "\\sa SARSA\n"
         "\n"
         "The difference between this algorithm and the original SARSA algorithm\n"
         "lies in the value used to approximate the value for the next timestep.\n"
         "In standard SARSA this value is directly taken as the current\n"
         "approximation of the value of the QFunction for the newly sampled state\n"
         "and the next action to be performed (the final 'SA' in SAR'SA').\n"
         "\n"
         "In Expected SARSA this value is instead replaced by the expected value\n"
         "for the newly sampled state, given the policy from which we will sample\n"
         "the next action. In this sense Expected SARSA is more similar to\n"
         "QLearning: where QLearning uses the max over the QFunction for the next\n"
         "state, Expected SARSA uses the future expectation over the current\n"
         "online policy.\n"
         "\n"
         "This reduces considerably the variance of the updates performed, which\n"
         "in turn allows to somewhat increase the learning rate for the method,\n"
         "which allows Expected SARSA to learn faster than simple SARSA. All\n"
         "guarantees of normal SARSA are maintained.", no_init}

        .def(init<QFunction &, const PolicyInterface &, optional<double, double>>(
                 "Basic constructor.\n"
                 "\n"
                 "Note that differently from normal SARSA, ExpectedSARSA does not\n"
                 "self-contain its own QFunction. This is because many policies\n"
                 "are implemented in terms of a QFunction continuously updated by\n"
                 "a method (e.g. QGreedyPolicy).\n"
                 "\n"
                 "At the same time ExpectedSARSA needs this policy in order to be\n"
                 "able to perform its expected value computation. In order to\n"
                 "avoid having a chicken and egg problem, ExpectedSARSA takes a\n"
                 "QFunction as parameter to allow the user to create it an use the\n"
                 "same one for both ExpectedSARSA and the policy.\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "@param qfun The QFunction underlying the ExpectedSARSA algorithm.\n"
                 "@param policy The policy used to select actions.\n"
                 "@param discount The discount of the underlying MDP model.\n"
                 "@param alpha The learning rate of the ExpectedSARSA method."
        , (arg("self"), "qfun", "policy", "discount", "alpha")))

        .def(init<QFunction &, const PolicyInterface &, const RLModel<Experience>&, optional<double>>(
                 "Basic constructor for RLModel.\n"
                 "\n"
                 "Note that differently from normal SARSA, ExpectedSARSA does not\n"
                 "self-contain its own QFunction. This is because many policies\n"
                 "are implemented in terms of a QFunction continuously updated by\n"
                 "a method (e.g. QGreedyPolicy).\n"
                 "\n"
                 "At the same time ExpectedSARSA needs this policy in order to be\n"
                 "able to perform its expected value computation. In order to\n"
                 "avoid having a chicken and egg problem, ExpectedSARSA takes a\n"
                 "QFunction as parameter to allow the user to create it an use the\n"
                 "same one for both ExpectedSARSA and the policy.\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the discount parameter from the supplied\n"
                 "model. It does not keep the reference, so if the discount needs\n"
                 "to change you'll need to update it here manually too.\n"
                 "\n"
                 "@param qfun The QFunction underlying the ExpectedSARSA algorithm.\n"
                 "@param policy The policy used to select actions.\n"
                 "@param model The MDP model that ExpectedSARSA will use as a base.\n"
                 "@param alpha The learning rate of the ExpectedSARSA method."
        , (arg("self"), "qfun", "policy", "model", "alpha")))

        .def(init<QFunction &, const PolicyInterface &, const SparseRLModel<SparseExperience>&, optional<double>>(
                 "Basic constructor for SparseRLModel.\n"
                 "\n"
                 "Note that differently from normal SARSA, ExpectedSARSA does not\n"
                 "self-contain its own QFunction. This is because many policies\n"
                 "are implemented in terms of a QFunction continuously updated by\n"
                 "a method (e.g. QGreedyPolicy).\n"
                 "\n"
                 "At the same time ExpectedSARSA needs this policy in order to be\n"
                 "able to perform its expected value computation. In order to\n"
                 "avoid having a chicken and egg problem, ExpectedSARSA takes a\n"
                 "QFunction as parameter to allow the user to create it an use the\n"
                 "same one for both ExpectedSARSA and the policy.\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the discount parameter from the supplied\n"
                 "model. It does not keep the reference, so if the discount needs\n"
                 "to change you'll need to update it here manually too.\n"
                 "\n"
                 "@param qfun The QFunction underlying the ExpectedSARSA algorithm.\n"
                 "@param policy The policy used to select actions.\n"
                 "@param model The MDP model that ExpectedSARSA will use as a base.\n"
                 "@param alpha The learning rate of the ExpectedSARSA method."
        , (arg("self"), "qfun", "policy", "model", "alpha")))

        .def(init<QFunction &, const PolicyInterface &, const Model&, optional<double>>(
                 "Basic constructor for Model.\n"
                 "\n"
                 "Note that differently from normal SARSA, ExpectedSARSA does not\n"
                 "self-contain its own QFunction. This is because many policies\n"
                 "are implemented in terms of a QFunction continuously updated by\n"
                 "a method (e.g. QGreedyPolicy).\n"
                 "\n"
                 "At the same time ExpectedSARSA needs this policy in order to be\n"
                 "able to perform its expected value computation. In order to\n"
                 "avoid having a chicken and egg problem, ExpectedSARSA takes a\n"
                 "QFunction as parameter to allow the user to create it an use the\n"
                 "same one for both ExpectedSARSA and the policy.\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the discount parameter from the supplied\n"
                 "model. It does not keep the reference, so if the discount needs\n"
                 "to change you'll need to update it here manually too.\n"
                 "\n"
                 "@param qfun The QFunction underlying the ExpectedSARSA algorithm.\n"
                 "@param policy The policy used to select actions.\n"
                 "@param model The MDP model that ExpectedSARSA will use as a base.\n"
                 "@param alpha The learning rate of the ExpectedSARSA method."
        , (arg("self"), "qfun", "policy", "model", "alpha")))

        .def(init<QFunction &, const PolicyInterface &, const SparseModel&, optional<double>>(
                 "Basic constructor for SparseModel.\n"
                 "\n"
                 "Note that differently from normal SARSA, ExpectedSARSA does not\n"
                 "self-contain its own QFunction. This is because many policies\n"
                 "are implemented in terms of a QFunction continuously updated by\n"
                 "a method (e.g. QGreedyPolicy).\n"
                 "\n"
                 "At the same time ExpectedSARSA needs this policy in order to be\n"
                 "able to perform its expected value computation. In order to\n"
                 "avoid having a chicken and egg problem, ExpectedSARSA takes a\n"
                 "QFunction as parameter to allow the user to create it an use the\n"
                 "same one for both ExpectedSARSA and the policy.\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the discount parameter from the supplied\n"
                 "model. It does not keep the reference, so if the discount needs\n"
                 "to change you'll need to update it here manually too.\n"
                 "\n"
                 "@param qfun The QFunction underlying the ExpectedSARSA algorithm.\n"
                 "@param policy The policy used to select actions.\n"
                 "@param model The MDP model that ExpectedSARSA will use as a base.\n"
                 "@param alpha The learning rate of the ExpectedSARSA method."
        , (arg("self"), "qfun", "policy", "model", "alpha")))

        .def("setLearningRate",             &ExpectedSARSA::setLearningRate,
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
                 "accordingly. The final behaviour of ExpectedSARSA is very\n"
                 "dependent on this parameter.\n"
                 "\n"
                 "The learning rate parameter must be > 0.0 and <= 1.0,\n"
                 "otherwise the function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param a The new learning rate parameter."
        , (arg("self"), "a"))

        .def("getLearningRate",             &ExpectedSARSA::getLearningRate,
                 "This function will return the current set learning rate parameter."
        , (arg("self")))

        .def("setDiscount",                 &ExpectedSARSA::setDiscount,
                 "This function sets the new discount parameter.\n"
                 "\n"
                 "The discount parameter controls the amount that future rewards are considered\n"
                 "by ExpectedSARSA. If 1, then any reward is the same, if obtained now or in a million\n"
                 "timesteps. Thus the algorithm will optimize overall reward accretion. When less\n"
                 "than 1, rewards obtained in the presents are valued more than future rewards.\n"
                 "\n"
                 "@param d The new discount factor."
        , (arg("self"), "d"))

        .def("getDiscount",                 &ExpectedSARSA::getDiscount,
                 "This function returns the currently set discount parameter."
        , (arg("self")))

        .def("stepUpdateQ",                 &ExpectedSARSA::stepUpdateQ,
                 "This function updates the internal QFunction using the discount set during construction.\n"
                 "\n"
                 "This function takes a single experience point and uses it to\n"
                 "update the QFunction. This is a very efficient method to\n"
                 "keep the QFunction up to date with the latest experience.\n"
                 "\n"
                 "Keep in mind that, since ExpectedSARSA needs to compute the\n"
                 "QFunction for the currently used policy, it needs to know\n"
                 "two consecutive state-action pairs, in order to correctly\n"
                 "relate how the policy acts from state to state.\n"
                 "\n"
                 "@param s The previous state.\n"
                 "@param a The action performed.\n"
                 "@param s1 The new state.\n"
                 "@param rew The reward obtained."
        , (arg("self"), "s", "a", "s1", "rew"))

        .def("getS",                        &ExpectedSARSA::getS,
                 "This function returns the number of states on which QLearning is working."
        , (arg("self")))

        .def("getA",                        &ExpectedSARSA::getA,
                 "This function returns the number of actions on which QLearning is working."
        , (arg("self")))

        .def("getQFunction",                &ExpectedSARSA::getQFunction, return_value_policy<reference_existing_object>(),
                 "This function returns a reference to the internal QFunction.\n"
                 "\n"
                 "The returned reference can be used to build Policies, for example\n"
                 "MDP::QGreedyPolicy."
        , (arg("self")))

        .def("getPolicy",                   &ExpectedSARSA::getPolicy, return_value_policy<reference_existing_object>(),
                 "This function returns a reference to the policy used by ExpectedSARSA."
        , (arg("self")));
}
