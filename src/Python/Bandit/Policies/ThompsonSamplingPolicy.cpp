#include <AIToolbox/Bandit/Policies/ThompsonSamplingPolicy.hpp>

#include <boost/python.hpp>

void exportBanditThompsonSamplingPolicy() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<ThompsonSamplingPolicy, bases<PolicyInterface>>{"ThompsonSamplingPolicy",

         "This class models a Thompson sampling policy.\n"
         "\n"
         "This class uses the Student-t distribution to model normally-distributed\n"
         "rewards with unknown mean and variance. As more experience is gained,\n"
         "each distribution becomes a Normal which models the mean of its\n"
         "respective arm.", no_init}

        .def(init<const QFunction &, const AIToolbox::Vector &, const std::vector<unsigned> &>(
                 "Basic constructor.\n"
                 "\n"
                 "@param q The QFunction to use as means for each actions.\n"
                 "@param M2s The sum over square distance from the mean.\n"
                 "@param counts The number of times each action has been tried before."
        , (arg("self"), "q", "M2s", "counts")));
}
