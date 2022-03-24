#include <AIToolbox/Bandit/Policies/ThompsonSamplingPolicy.hpp>

#include <boost/python.hpp>

void exportBanditThompsonSamplingPolicy() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<ThompsonSamplingPolicy, bases<PolicyInterface>>{"ThompsonSamplingPolicy",

         "This class implements a Thompson sampling policy.\n"
         "\n"
         "This class uses the Student-t distribution to model normally-distributed\n"
         "rewards with unknown mean and variance. As more experience is gained,\n"
         "each distribution becomes a Normal which models the mean of its\n"
         "respective arm.", no_init}

        .def(init<const Experience &>(
                 "Basic constructor.\n"
                 "\n"
                 "@param exp The Experience we learn from."
        , (arg("self"), "exp")));
}
