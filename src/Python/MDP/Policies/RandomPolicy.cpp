#include <AIToolbox/MDP/Policies/RandomPolicy.hpp>

#include <boost/python.hpp>

void exportMDPRandomPolicy() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<RandomPolicy, bases<PolicyInterface>>{"Policy",

         "This class represents an MDP Random Policy.\n"
         "\n"
         "This class simply returns a random action when it is polled.", no_init}

        .def(init<size_t, size_t>(
                 "Basic constructor.\n"
                 "\n"
                 "@param s The number of states of the world.\n"
                 "@param a The number of actions available to the agent."
        , (arg("self"), "s", "a")));
}
