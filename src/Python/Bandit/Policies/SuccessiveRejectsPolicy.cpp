#include <AIToolbox/Bandit/Policies/SuccessiveRejectsPolicy.hpp>

#include <boost/python.hpp>

void exportBanditSuccessiveRejectsPolicy() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<SuccessiveRejectsPolicy, bases<PolicyInterface>>{"SuccessiveRejectsPolicy",

         "This class implements the successive rejects algorithm.\n"
         "\n"
         "The successive rejects (SR) algorithm is a budget-based pure exploration\n"
         "algorithm. Its goal is to simply recommend the best possible action\n"
         "after its budget of pulls has been exhausted. The reward accumulated\n"
         "during the exploration phase is irrelevant to the algorithm itself,\n"
         "which is only focused on optimizing the quality of the final\n"
         "recommendation.\n"
         "\n"
         "The way SR works is to split the available budget into phases. During\n"
         "each phase, each arm is pulled a certain (nK_) number of times, which\n"
         "depends on the current phase. After these pulls, the arm with the lowest\n"
         "empirical mean is removed from the pool of arms to be evaluated.\n"
         "\n"
         "The algorithm keeps removing arms from the pool until a single arm\n"
         "remains, which corresponds to the final recommended arm.", no_init}

        .def(init<const Experience &, unsigned>(
                 "Basic constructor.\n"
                 "\n"
                 "@param exp The Experience we learn from.\n"
                 "@param budget The overall pull budget for the exploration."
        , (arg("self"), "exp", "budget")))

        .def("stepUpdateQ", &SuccessiveRejectsPolicy::stepUpdateQ,
                 "This function updates the current phase, nK_, and prunes actions from the pool.\n"
                 "\n"
                 "This function must be called each timestep after the Experience has been updated.\n"
                 "\n"
                 "If needed, it will trigger pulling the next action in sequence.\n"
                 "If all actions have been pulled nK_ times, it will increase the\n"
                 "current phase, update nK_ and perform the appropriate pruning\n"
                 "using the current reward estimates contained in the underlying\n"
                 "Experience."
        , (arg("self")))

        .def("canRecommendAction", &SuccessiveRejectsPolicy::canRecommendAction,
                "This function returns whether a single action remains in the pool."
        , (arg("self")))

        .def("recommendAction", &SuccessiveRejectsPolicy::recommendAction,
                "If the pool has a single element, this function returns the best estimated action after the SR exploration process."
        , (arg("self")))

        .def("getCurrentPhase", &SuccessiveRejectsPolicy::getCurrentPhase,
                 "This function returns the current phase.\n"
                 "\n"
                 "Note that if the exploration process is ended, the current phase will be equal to the number of actions."
        , (arg("self")))

        .def("getCurrentNk", &SuccessiveRejectsPolicy::getCurrentNk,
                 "This function returns the nK_ for the current phase."
        , (arg("self")))

        .def("getExperience", &SuccessiveRejectsPolicy::getExperience, return_internal_reference<>(),
                "This function returns a reference to the underlying Experience we use."
        , (arg("self")));
}
