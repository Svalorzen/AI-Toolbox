#include <AIToolbox/MDP/Algorithms/SARSAL.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportMDPSARSAL() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    class_<SARSAL>{"SARSAL",

         "This class represents the SARSAL algorithm.\n"
         "\n"
         "This algorithms adds eligibility traces to the SARSA algorithm.\n"
         "\n"
         "\\sa SARSA\n"
         "\n"
         "In order to more effectively use the data obtained, SARSAL keeps a list\n"
         "of previously visited state/action pairs, which are updated together\n"
         "with the last experienced transition. The updates all use the same\n"
         "value, with the difference that state/action pairs experienced more in\n"
         "the past are updated less (by discount*lambda per each previous\n"
         "timestep). Once this reducing coefficient falls below a certain\n"
         "threshold, the old state/action pair is forgotten and not updated\n"
         "anymore. If instead the pair is visited again, the coefficient is once\n"
         "again increased.\n"
         "\n"
         "The idea is to be able to give credit to past actions for current reward\n"
         "in an efficient manner. This reduces the amount of data needed in order\n"
         "to backpropagate rewards, and allows SARSAL to learn faster.\n"
         "\n"
         "This particular version of the algorithm implements capped traces: every\n"
         "time an action/state pair is witnessed, its eligibility trace is reset\n"
         "to 1.0. This avoids potentially diverging values which can happen with\n"
         "the normal eligibility traces.", no_init}

        .def(init<size_t, size_t, optional<double, double, double, double>>(
                 "Basic constructor.\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "@param S The state space of the underlying model.\n"
                 "@param A The action space of the underlying model.\n"
                 "@param discount The discount of the underlying model.\n"
                 "@param alpha The learning rate of the SARSAL method.\n"
                 "@param lambda The lambda parameter for the eligibility traces.\n"
                 "@param tolerance The cutoff point for eligibility traces."
        , (arg("self"), "S", "A", "discount", "alpha", "lambda", "tolerance")))

        .def(init<const RLModel<Experience>&, optional<double, double, double>>(
                 "Basic constructor for RLModel.\n"
                 "\n"
                 "The learning rate must be > 0.0 and <= 1.0, otherwise the\n"
                 "constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not keep the reference, so if the\n"
                 "discount needs to change you'll need to update it here manually\n"
                 "too.\n"
                 "\n"
                 "@param model The MDP model that SARSAL will use as a base.\n"
                 "@param alpha The learning rate of the SARSAL method.\n"
                 "@param lambda The lambda parameter for the eligibility traces.\n"
                 "@param tolerance The cutoff point for eligibility traces."
        , (arg("self"), "model", "alpha", "lambda", "tolerance")))

        .def(init<const SparseRLModel<SparseExperience>&, optional<double, double, double>>(
                 "Basic constructor for SparseRLModel.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not keep the reference, so if the\n"
                 "discount needs to change you'll need to update it here manually\n"
                 "too.\n"
                 "\n"
                 "@param model The MDP model that SARSAL will use as a base.\n"
                 "@param alpha The learning rate of the SARSAL method.\n"
                 "@param lambda The lambda parameter for the eligibility traces.\n"
                 "@param tolerance The cutoff point for eligibility traces."
        , (arg("self"), "model", "alpha", "lambda", "tolerance")))

        .def(init<const Model&, optional<double, double, double>>(
                 "Basic constructor for Model.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not keep the reference, so if the\n"
                 "discount needs to change you'll need to update it here manually\n"
                 "too.\n"
                 "\n"
                 "@param model The MDP model that SARSAL will use as a base.\n"
                 "@param alpha The learning rate of the SARSAL method.\n"
                 "@param lambda The lambda parameter for the eligibility traces.\n"
                 "@param tolerance The cutoff point for eligibility traces."
        , (arg("self"), "model", "alpha", "lambda", "tolerance")))

        .def(init<const SparseModel&, optional<double, double, double>>(
                 "Basic constructor for SparseModel.\n"
                 "\n"
                 "This constructor copies the S and A and discount parameters from\n"
                 "the supplied model. It does not keep the reference, so if the\n"
                 "discount needs to change you'll need to update it here manually\n"
                 "too.\n"
                 "\n"
                 "@param model The MDP model that SARSAL will use as a base.\n"
                 "@param alpha The learning rate of the SARSAL method.\n"
                 "@param lambda The lambda parameter for the eligibility traces.\n"
                 "@param tolerance The cutoff point for eligibility traces."
        , (arg("self"), "model", "alpha", "lambda", "tolerance")))

        .def("setLearningRate",             &SARSAL::setLearningRate,
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
                 "accordingly. The final behaviour of SARSAL is very\n"
                 "dependent on this parameter.\n"
                 "\n"
                 "The learning rate parameter must be > 0.0 and <= 1.0,\n"
                 "otherwise the function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param a The new learning rate parameter."
        , (arg("self"), "a"))

        .def("getLearningRate",             &SARSAL::getLearningRate,
                 "This function will return the current set learning rate parameter."
        , (arg("self")))

        .def("setDiscount",                 &SARSAL::setDiscount,
                 "This function sets the new discount parameter.\n"
                 "\n"
                 "The discount parameter controls the amount that future rewards are considered\n"
                 "by SARSAL. If 1, then any reward is the same, if obtained now or in a million\n"
                 "timesteps. Thus the algorithm will optimize overall reward accretion. When less\n"
                 "than 1, rewards obtained in the presents are valued more than future rewards.\n"
                 "\n"
                 "@param d The new discount factor."
        , (arg("self"), "d"))

        .def("getDiscount",                 &SARSAL::getDiscount,
                 "This function returns the currently set discount parameter."
        , (arg("self")))

        .def("setLambda",                   &SARSAL::setLambda,
                 "This function sets the new lambda parameter.\n"
                 "\n"
                 "This parameter determines how much to decrease updates for each\n"
                 "timestep in the past. If set to zero, SARSAL effectively becomes\n"
                 "equivalent to SARSA, as no backpropagation will be performed. If\n"
                 "set to 1 it will result in a method similar to Monte Carlo\n"
                 "sampling, where rewards are backed up from the end to the\n"
                 "beginning of the episode (of course still dependent on the\n"
                 "discount of the model).\n"
                 "\n"
                 "The lambda parameter must be >= 0.0 and <= 1.0, otherwise the\n"
                 "function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param l The new lambda parameter."
        , (arg("self"), "lambda"))

        .def("getLambda",                   &SARSAL::getLambda,
                 "This function returns the currently set lambda parameter."
        , (arg("self")))

        .def("setTolerance",                &SARSAL::setTolerance,
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

        .def("getTolerance",                &SARSAL::getTolerance,
                 "This function returns the currently set trace cutoff parameter."
        , (arg("self")))

        .def("stepUpdateQ",                 &SARSAL::stepUpdateQ,
                 "This function updates the internal QFunction using the discount set during construction.\n"
                 "\n"
                 "This function takes a single experience point and uses it to\n"
                 "update the QFunction. This is a very efficient method to\n"
                 "keep the QFunction up to date with the latest experience.\n"
                 "\n"
                 "Keep in mind that, since SARSAL needs to compute the\n"
                 "QFunction for the currently used policy, it needs to know\n"
                 "two consecutive state-action pairs, in order to correctly\n"
                 "relate how the policy acts from state to state.\n"
                 "\n"
                 "@param s The previous state.\n"
                 "@param a The action performed.\n"
                 "@param s1 The new state.\n"
                 "@param a1 The action performed in the new state.\n"
                 "@param rew The reward obtained."
        , (arg("self"), "s", "a", "s1", "a1", "rew"))

        .def("getS",                        &SARSAL::getS,
                 "This function returns the number of states on which QLearning is working."
        , (arg("self")))

        .def("getA",                        &SARSAL::getA,
                 "This function returns the number of actions on which QLearning is working."
        , (arg("self")))

        .def("getQFunction",                &SARSAL::getQFunction, return_internal_reference<>(),
                 "This function returns a reference to the internal QFunction.\n"
                 "\n"
                 "The returned reference can be used to build Policies, for example\n"
                 "MDP::QGreedyPolicy."
        , (arg("self")))

        .def("getTraces",                   &SARSAL::getTraces, return_internal_reference<>(),
                 "This function returns the currently set traces."
        , (arg("self")));
}
