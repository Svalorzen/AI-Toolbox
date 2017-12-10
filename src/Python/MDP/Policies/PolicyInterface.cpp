#include <AIToolbox/MDP/Policies/PolicyInterface.hpp>

#include <boost/python.hpp>

void exportMDPPolicyInterface() {
    using namespace AIToolbox;
    using namespace boost::python;

    class_<MDP::PolicyInterface, boost::noncopyable>{"PolicyInterface",

         "This class represents the base interface for policies in MDPs.\n"
         "\n"
         "This class represents an interface that all policies must conform to.\n"
         "The interface is generic as different methods may have very different\n"
         "ways to store and compute policies, and this interface simply asks\n"
         "for a way to sample them.\n"
         "\n"
         "In the case of MDPs, the class works using integer states, which\n"
         "represent the discrete states from which we are sampling.", no_init}

        .def("sampleAction",            &MDP::PolicyInterface::sampleAction,
             "This function chooses a random action for state s, following the policy distribution.\n"
             "\n"
             "@param s The sampled state of the policy.\n"
             "\n"
             "@return The chosen action."
        , (arg("self"), "s"))

        .def("getActionProbability",    &MDP::PolicyInterface::getActionProbability,
             "This function returns the probability of taking the specified action in the specified state.\n"
             "\n"
             "@param s The selected state.\n"
             "@param a The selected action.\n"
             "\n"
             "@return The probability of taking the selected action in the specified state.\n"
        , (arg("self"), "s", "a"));
}
