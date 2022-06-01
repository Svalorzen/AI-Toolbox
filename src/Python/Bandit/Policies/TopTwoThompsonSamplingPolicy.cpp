#include <AIToolbox/Bandit/Policies/TopTwoThompsonSamplingPolicy.hpp>

#include <boost/python.hpp>

void exportBanditTopTwoThompsonSamplingPolicy() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<TopTwoThompsonSamplingPolicy, bases<PolicyInterface>>{"TopTwoThompsonSamplingPolicy",

         "This class implements the top-two Thompson sampling policy.\n"
         "\n"
         "This class uses the Student-t distribution to model normally-distributed\n"
         "rewards with unknown mean and variance. As more experience is gained,\n"
         "each distribution becomes a Normal which models the mean of its\n"
         "respective arm.\n"
         "\n"
         "The top-two Thompson sampling policy is designed to be used in a pure\n"
         "exploration setting. In other words, we wish to discover the best arm in\n"
         "the shortest possible time, without the need to minimize regret while\n"
         "doing so. This last part is the key difference to many bandit\n"
         "algorithms, that try to exploit their knowledge more and more as time\n"
         "goes on.\n"
         "\n"
         "The way this works is by focusing arm pulls on the currently estimated\n"
         "top two arms, since those are the most likely to contend for the 'title'\n"
         "of best arm. The two top arms are estimated using Thompson sampling. We\n"
         "first sample a first best action, and then, if needed, we keep sampling\n"
         "until a new, different best action is sampled.\n"
         "\n"
         "We either take the first action sampled with probability beta, or the\n"
         "other with probability 1-beta.", no_init}

        .def(init<const Experience &, double>(
                 "Basic constructor.\n"
                 "\n"
                 "@param exp The Experience we learn from.\n"
                 "@param beta The probability of playing the first sampled best action instead of the second sampled best."
        , (arg("self"), "exp", "beta")))

        .def("recommendAction", &TopTwoThompsonSamplingPolicy::recommendAction,
                "This function returns the most likely best action until this point."
        , (arg("self")))

        .def("getExperience", &TopTwoThompsonSamplingPolicy::getExperience, return_internal_reference<>(),
                "This function returns a reference to the underlying Experience we use."
        , (arg("self")));
}
