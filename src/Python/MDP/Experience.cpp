#include <AIToolbox/MDP/Experience.hpp>

#include <boost/python.hpp>

void exportMDPExperience() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<Experience>{"Experience",

         "This class keeps track of registered events and rewards.\n"
         "\n"
         "This class is a simple logger of events. It keeps track of both\n"
         "the number of times a particular transition has happened, and the\n"
         "total reward gained in any particular transition. However, it\n"
         "does not record each event separately (i.e. you can't extract\n"
         "the results of a particular transition in the past).", no_init}

        .def(init<size_t, size_t>(
                 "Basic constructor.\n"
                 "\n"
                 "@param s The number of states of the world.\n"
                 "@param a The number of actions available to the agent."
        , (arg("self"), "s", "a")))

        .def("record",          &Experience::record,
                 "This function adds a new event to the recordings.\n"
                 "\n"
                 "@param s     Old state.\n"
                 "@param a     Performed action.\n"
                 "@param s1    New state.\n"
                 "@param rew   Obtained reward."
        , (arg("self"), "s", "a", "s1", "rew"))

        .def("reset",           &Experience::reset,
                "This function resets all experienced rewards and transitions."
        , (arg("self")))

        .def("setVisits",       &Experience::setVisits<std::vector<std::vector<std::vector<int>>>>,
                 "Compatibility setter.\n"
                 "\n"
                 "This function takes an arbitrary three dimensional\n"
                 "containers and tries to copy its contents into the\n"
                 "visits table.\n"
                 "\n"
                 "Currently the Python wrappings support reading through native\n"
                 "3d Python arrays (so [][][]). As long as the dimensions are\n"
                 "correct it should work.\n"
                 "\n"
                 "@param v The external visits container.\n"
        , (arg("self"), "v"))

        .def("setRewards",       &Experience::setRewards<std::vector<std::vector<std::vector<double>>>>,
                 "Compatibility setter.\n"
                 "\n"
                 "This function takes an arbitrary three dimensional\n"
                 "container and tries to copy its contents into the\n"
                 "rewards matrix.\n"
                 "\n"
                 "Currently the Python wrappings support reading through native\n"
                 "3d Python arrays (so [][][]). As long as the dimensions are\n"
                 "correct it should work.\n"
                 "\n"
                 "@param v The external visits container.\n"
        , (arg("self"), "v"))

        .def("getVisits",       &Experience::getVisits,
                 "This function returns the current recorded visits for a transitions.\n"
                 "\n"
                 "@param s     Old state.\n"
                 "@param a     Performed action.\n"
                 "@param s1    New state."
        , (arg("self"), "s", "a", "s1"))

        .def("getVisitsSum",    &Experience::getVisitsSum,
                 "This function returns the number of transitions recorded that start with the specified state and action.\n"
                 "\n"
                 "@param s     The initial state.\n"
                 "@param a     Performed action.\n"
                 "\n"
                 "@return The total number of transitions that start with the specified state-action pair."
        , (arg("self"), "s", "a"))

        .def("getReward",       &Experience::getReward,
                 "This function returns the cumulative rewards obtained from a specific transition.\n"
                 "\n"
                 "@param s     Old state.\n"
                 "@param a     Performed action.\n"
                 "@param s1    New state."
        , (arg("self"), "s", "a", "s1"))

        .def("getRewardSum",    &Experience::getRewardSum,
                 "This function returns the total reward obtained from transitions that start with the specified state and action.\n"
                 "\n"
                 "@param s     The initial state.\n"
                 "@param a     Performed action.\n"
                 "\n"
                 "@return The total number of transitions that start with the specified state-action pair."
        , (arg("self"), "s", "a"))

        .def("getS",            &Experience::getS,
                 "This function returns the number of states of the world.\n"
                 "\n"
                 "@return The total number of states."
        , (arg("self")))

        .def("getA",            &Experience::getA,
                 "This function returns the number of available actions to the agent.\n"
                 "\n"
                 "@return The total number of actions."
        , (arg("self")));
}
