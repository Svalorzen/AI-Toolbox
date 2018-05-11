#include <AIToolbox/Bandit/Policies/ThompsonSamplingPolicy.hpp>

#include <boost/python.hpp>

void exportBanditThompsonSamplingPolicy() {
    using namespace AIToolbox;
    using namespace boost::python;

    class_<Bandit::ThompsonSamplingPolicy, bases<Bandit::PolicyInterface>>{"ThompsonSamplingPolicy",

         "This class models a Thompson sampling policy.\n"
         "\n"
         "This class keeps a record of the rewards obtained by each action, and\n"
         "chooses them with a stochastic policy which is proportional to the\n"
         "goodness of each action.\n"
         "\n"
         "It uses the Normal distribution in order to estimate its certainty about\n"
         "each arm average reward. Thus, each arm is estimated through a Normal\n"
         "distribution centered on the average for the arm, with decreasing\n"
         "variance as more experience is gathered.\n"
         "\n"
         "Note that this class assumes that the reward obtained is normalized into\n"
         "a [0,1] range (which it does not check).\n"
         "\n"
         "The usage of the Normal distribution best matches a Normally distributed\n"
         "reward. Another implementation (not provided here) uses Beta\n"
         "distributions to handle Bernoulli distributed rewards.", no_init}

        .def(init<size_t>(
                 "Basic constructor.\n"
                 "\n"
                 "@param A The size of the action space."
        , (arg("self"), "A")))

        .def("stepUpdateP",             &Bandit::ThompsonSamplingPolicy::stepUpdateP,
                 "This function updates the policy based on the result of the action.\n"
                 "\n"
                 "We simply keep a rolling average for each action, which we\n"
                 "update here. Each average and count will then be used as\n"
                 "parameters for the Normal distribution used to decide which\n"
                 "action to sample later.\n"
                 "\n"
                 "Note that we expect a normalized reward here in order to\n"
                 "easily compare the various actions during Normal sampling.\n"
                 "\n"
                 "@param a The action taken.\n"
                 "@param r The reward obtained, in a [0,1] range."
        , (arg("self"), "a", "r"));
}

