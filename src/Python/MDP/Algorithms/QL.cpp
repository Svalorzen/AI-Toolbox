#include <AIToolbox/MDP/Algorithms/QL.hpp>

#include <boost/python.hpp>

void exportMDPQL() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    class_<QL>{"QL",

         "This class implements off-policy evaluation via Q(lambda)."
         "\n"
         "This algorithm is the off-policy equivalent of SARSAL. It scales traces\n"
         "using the lambda parameter, but is able to work in an off-line manner.\n"
         "\n"
         "Unfortunately, as it does not take into account the discrepancy between\n"
         "behaviour and target policies, it tends to work only if the two policies\n"
         "are similar.\n"
         "\n"
         "Note that even if the trace discount does not take into account the\n"
         "target policy, the error update is still computed using the target, and\n"
         "that is why the method works and does not just compute the value of the\n"
         "current behaviour policy.\n"
         "\n"
         "This method behaves as an inefficient QLearning if you set the lambda\n"
         "parameter to zero (effectively cutting all traces), and the epsilon\n"
         "parameter to zero (forcing a perfectly greedy target policy).", no_init}

        .def(init<size_t, size_t, optional<double, double, double, double, double>>(
                 "Basic constructor.\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "@param S The state space of the underlying model.\n"
                 "@param A The action space of the underlying model.\n"
                 "@param discount The discount of the underlying model.\n"
                 "@param alpha The learning rate of the QL method.\n"
                 "@param lambda The lambda parameter for the eligibility traces.\n"
                 "@param tolerance The cutoff point for eligibility traces.\n"
                 "@param epsilon The epsilon of the implied target greedy epsilon policy."
        , (arg("self"), "S", "A", "discount", "alpha", "lambda", "tolerance", "epsilon")))

        .def("setLearningRate",             &QL::setLearningRate,
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
                 "accordingly. The final behaviour of QL is very\n"
                 "dependent on this parameter.\n"
                 "\n"
                 "The learning rate parameter must be > 0.0 and <= 1.0,\n"
                 "otherwise the function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param a The new learning rate parameter."
        , (arg("self"), "a"))

        .def("getLearningRate",             &QL::getLearningRate,
                 "This function will return the current set learning rate parameter."
        , (arg("self")))

        .def("setDiscount",                 &QL::setDiscount,
                 "This function sets the new discount parameter.\n"
                 "\n"
                 "The discount parameter controls the amount that future rewards are considered\n"
                 "by QL. If 1, then any reward is the same, if obtained now or in a million\n"
                 "timesteps. Thus the algorithm will optimize overall reward accretion. When less\n"
                 "than 1, rewards obtained in the presents are valued more than future rewards.\n"
                 "\n"
                 "@param d The new discount factor."
        , (arg("self"), "d"))

        .def("getDiscount",                 &QL::getDiscount,
                 "This function returns the currently set discount parameter."
        , (arg("self")))

        .def("setLambda",                   &QL::setLambda,
                 "This function sets the new lambda parameter.\n"
                 "\n"
                 "The lambda parameter must be >= 0.0 and <= 1.0, otherwise the\n"
                 "function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param l The new lambda parameter."
        , (arg("self"), "lambda"))

        .def("getLambda",                   &QL::getLambda,
                 "This function returns the currently set lambda parameter."
        , (arg("self")))

        .def("setTolerance",                &QL::setTolerance,
                 "This function sets the trace cutoff parameter.\n"
                 "\n"
                 "This parameter determines when a trace is removed, as its\n"
                 "coefficient has become too small to bother updating its value.\n"
                 "\n"
                 "Note that the trace cutoff is performed on the overall\n"
                 "discount*lambda value, and not only on lambda. So this parameter\n"
                 "is useful even when lambda is 1.\n"
                 "\n"
                 "@param t The new trace cutoff value."
        , (arg("self"), "t"))

        .def("getTolerance",                &QL::getTolerance,
                 "This function returns the currently set trace cutoff parameter."
        , (arg("self")))

        .def("stepUpdateQ",                 &QL::stepUpdateQ,
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

        .def("getS",                        &QL::getS,
                 "This function returns the number of states on which QLearning is working."
        , (arg("self")))

        .def("getA",                        &QL::getA,
                 "This function returns the number of actions on which QLearning is working."
        , (arg("self")))

        .def("getQFunction",                &QL::getQFunction, return_internal_reference<>(),
                 "This function returns a reference to the internal QFunction.\n"
                 "\n"
                 "The returned reference can be used to build Policies, for example\n"
                 "MDP::QGreedyPolicy."
        , (arg("self")))

        .def("getTraces",                   &QL::getTraces, return_internal_reference<>(),
                 "This function returns the currently set traces."
        , (arg("self")));
}
