#include <AIToolbox/MDP/Algorithms/HystereticQLearning.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportMDPHystereticQLearning() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    class_<HystereticQLearning>{"HystereticQLearning",

         "This class represents the Hysteretic QLearning algorithm.\n"
         "\n"
         "This algorithm is a very simple but powerful way to learn the\n"
         "optimal QFunction for an MDP model, where the transition and reward\n"
         "functions are unknown. It works in an offline fashion, meaning that\n"
         "it can be used even if the policy that the agent is currently using\n"
         "is not the optimal one, or is different by the one currently\n"
         "specified by the HystereticQLearning QFunction.\n"
         "\n"
         "The algorithm functions quite like the normal QLearning algorithm, with\n"
         "a small difference: it has an additional learning parameter, beta.\n"
         "\n"
         "One of the learning parameters (alpha) is used when the change to the\n"
         "underlying QFunction is positive. The other (beta), which should be kept\n"
         "lower than alpha, is used when the change is negative.\n"
         "\n"
         "This is useful when using QLearning for multi-agent RL where each agent\n"
         "is independent. A multi-agent environment is non-stationary from the\n"
         "point of view of a single agent, which is disruptive for normal\n"
         "QLearning and generally prevents it to learn to coordinate with the\n"
         "other agents well.\n"
         "\n"
         "By assigning a higher learning parameter to transitions resulting in a\n"
         "positive feedback, the agent insulates itself from bad results which\n"
         "happen when the other agents take exploratory actions.\n"
         "\n"
         "Bad results are still guaranteed to be discovered, since the learning\n"
         "parameter is still greater than zero, but the algorithm tries to focus\n"
         "on the good things rather than the bad.\n"
         "\n"
         "If the beta parameter is equal to the alpha, this becomes standard\n"
         "QLearning. When the beta parameter is zero, the algorithm becomes\n"
         "equivalent to Distributed QLearning.", no_init}

        .def(init<size_t, size_t, optional<double, double, double>>(
                 "Basic constructor.\n"
                 "\n"
                 "The alpha learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "The beta learning rate must be >= 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument. It can be zero.\n"
                 "\n"
                 "Keep in mind that the beta parameter should be lower than the\n"
                 "alpha parameter, although this is not enforced.\n"
                 "\n"
                 "@param S The size of the state space.\n"
                 "@param A The size of the action space.\n"
                 "@param discount The discount to use when learning.\n"
                 "@param alpha The learning rate for positive updates.\n"
                 "@param beta The learning rate for negative updates."
        , (arg("self"), "S", "A", "discount", "alpha", "beta")))

        .def(init<const RLModel<Experience>&, optional<double, double>>(
                 "Basic constructor for RLModel.\n"
                 "\n"
                 "The alpha learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "The beta learning rate must be >= 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument. It can be zero.\n"
                 "\n"
                 "Keep in mind that the beta parameter should be lower than the\n"
                 "alpha parameter, although this is not enforced.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not keep the reference, so if the\n"
                 "discount needs to change you'll need to update it here manually\n"
                 "too.\n"
                 "\n"
                 "@param model The MDP model that HystereticQLearning will use as a base.\n"
                 "@param alpha The learning rate of the HystereticQLearning method.\n"
                 "@param beta The learning rate for negative updates."
        , (arg("self"), "model", "alpha", "beta")))

        .def(init<const SparseRLModel<SparseExperience>&, optional<double, double>>(
                 "Basic constructor for SparseRLModel.\n"
                 "\n"
                 "The alpha learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "The beta learning rate must be >= 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument. It can be zero.\n"
                 "\n"
                 "Keep in mind that the beta parameter should be lower than the\n"
                 "alpha parameter, although this is not enforced.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not keep the reference, so if the\n"
                 "discount needs to change you'll need to update it here manually\n"
                 "too.\n"
                 "\n"
                 "@param model The MDP model that HystereticQLearning will use as a base.\n"
                 "@param alpha The learning rate of the HystereticQLearning method.\n"
                 "@param beta The learning rate for negative updates."
        , (arg("self"), "model", "alpha", "beta")))

        .def(init<const Model&, optional<double, double>>(
                 "Basic constructor for Model.\n"
                 "\n"
                 "The alpha learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "The beta learning rate must be >= 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument. It can be zero.\n"
                 "\n"
                 "Keep in mind that the beta parameter should be lower than the\n"
                 "alpha parameter, although this is not enforced.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not keep the reference, so if the\n"
                 "discount needs to change you'll need to update it here manually\n"
                 "too.\n"
                 "\n"
                 "@param model The MDP model that HystereticQLearning will use as a base.\n"
                 "@param alpha The learning rate of the HystereticQLearning method.\n"
                 "@param beta The learning rate for negative updates."
        , (arg("self"), "model", "alpha", "beta")))

        .def(init<const SparseModel&, optional<double, double>>(
                 "Basic constructor for SparseModel.\n"
                 "\n"
                 "The alpha learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "The beta learning rate must be >= 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument. It can be zero.\n"
                 "\n"
                 "Keep in mind that the beta parameter should be lower than the\n"
                 "alpha parameter, although this is not enforced.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not keep the reference, so if the\n"
                 "discount needs to change you'll need to update it here manually\n"
                 "too.\n"
                 "\n"
                 "@param model The MDP model that HystereticQLearning will use as a base.\n"
                 "@param alpha The learning rate of the HystereticQLearning method.\n"
                 "@param beta The learning rate for negative updates."
        , (arg("self"), "model", "alpha", "beta")))

        .def("setPositiveLearningRate",             &HystereticQLearning::setPositiveLearningRate,
                 "This function sets the learning rate parameter for positive updates.\n"
                 "\n"
                 "The learning parameter determines the speed at which the\n"
                 "QFunction is modified with respect to new data, when updates are\n"
                 "positive.\n"
                 "\n"
                 "The learning rate parameter must be > 0.0 and <= 1.0,\n"
                 "otherwise the function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param a The new learning rate parameter for positive updates."
        , (arg("self"), "a"))

        .def("getPositiveLearningRate",             &HystereticQLearning::getPositiveLearningRate,
                 "This function will return the currently set positive learning rate parameter."
        , (arg("self")))

        .def("setNegativeLearningRate",             &HystereticQLearning::setNegativeLearningRate,
                 "This function sets the learning rate parameter for negative updates.\n"
                 "\n"
                 "The learning parameter determines the speed at which the\n"
                 "QFunction is modified with respect to new data, when updates are\n"
                 "negative.\n"
                 "\n"
                 "Note that this parameter can be zero.\n"
                 "\n"
                 "The learning rate parameter must be >= 0.0 and <= 1.0,\n"
                 "otherwise the function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param b The new learning rate parameter for negative updates."
        , (arg("self"), "b"))

        .def("getNegativeLearningRate",             &HystereticQLearning::getNegativeLearningRate,
                 "This function will return the currently set negative learning rate parameter."
        , (arg("self")))

        .def("setDiscount",                 &HystereticQLearning::setDiscount,
                 "This function sets the new discount parameter.\n"
                 "\n"
                 "The discount parameter controls the amount that future rewards\n"
                 "are considered by HystereticQLearning. If 1, then any reward is\n"
                 "the same, if obtained now or in a million timesteps. Thus the\n"
                 "algorithm will optimize overall reward accretion. When less than\n"
                 "1, rewards obtained in the presents are valued more than future\n"
                 "rewards.\n"
                 "\n"
                 "@param d The new discount factor."
        , (arg("self"), "d"))

        .def("getDiscount",                 &HystereticQLearning::getDiscount,
                 "This function returns the currently set discount parameter."
        , (arg("self")))

        .def("stepUpdateQ",                 &HystereticQLearning::stepUpdateQ,
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

        .def("getS",                        &HystereticQLearning::getS,
                 "This function returns the number of states on which HystereticQLearning is working."
        , (arg("self")))

        .def("getA",                        &HystereticQLearning::getA,
                 "This function returns the number of actions on which HystereticQLearning is working."
        , (arg("self")))

        .def("getQFunction",                &HystereticQLearning::getQFunction, return_internal_reference<>(),
                 "This function returns a reference to the internal QFunction.\n"
                 "\n"
                 "The returned reference can be used to build Policies, for example\n"
                 "MDP::QGreedyPolicy."
        , (arg("self")));
}
