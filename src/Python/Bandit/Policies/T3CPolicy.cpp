#include <AIToolbox/Bandit/Policies/T3CPolicy.hpp>

#include <boost/python.hpp>

void exportBanditT3CPolicy() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<T3CPolicy, bases<PolicyInterface>>{"T3CPolicy",

         "This class implements the T3C sampling policy.\n"
         "\n"
         "This class assumes that the rewards of all bandit arms are normally\n"
         "distributed, with all arms having the same variance.\n"
         "\n"
         "T3C was designed as a replacement for TopTwoThompsonSamplingPolicy. The\n"
         "main idea is that, when we want to pull the estimated *second* best arm,\n"
         "instead of having to resample the arm means until a new unique contender\n"
         "appears, we can deterministically compute that contender using a measure\n"
         "of distance between the distributions of the arms.\n"
         "\n"
         "This allows the algorithm to keep the computational costs low even after\n"
         "many pulls, while TopTwoThompsonSamplingPolicy tends to degrade in\n"
         "performance as time passes (as resampling is less and less likely to\n"
         "generate a unique second best contender).", no_init}

        .def(init<const Experience &, double, double>(
                 "Basic constructor.\n"
                 "\n"
                 "@param exp The Experience we learn from.\n"
                 "@param beta The probability of playing the first sampled best action instead of the second sampled best.\n"
                 "@param var The known variance of all arms."
        , (arg("self"), "exp", "beta", "var")))

        .def("recommendAction", &T3CPolicy::recommendAction,
                "This function returns the most likely best action until this point."
        , (arg("self")))

        .def("getExperience", &T3CPolicy::getExperience, return_internal_reference<>(),
                "This function returns a reference to the underlying Experience we use."
        , (arg("self")));
}
