#include <AIToolbox/Bandit/Policies/ThompsonSamplingPolicy.hpp>

#include <boost/python.hpp>

void exportBanditThompsonSamplingPolicy() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<ThompsonSamplingPolicy, bases<PolicyInterface>>{"ThompsonSamplingPolicy",

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

        .def(init<const QFunction &, const std::vector<unsigned> &>(
                 "Basic constructor.\n"
                 "\n"
                 "@param q The QFunction to use for the means\n"
                 "@param counts The counts for each tried action"
        , (arg("self"), "q", "counts")));
}

