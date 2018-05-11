#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>

#include <boost/python.hpp>

void exportBanditPolicyInterface() {
    using namespace AIToolbox;
    using namespace boost::python;

    class_<Bandit::PolicyInterface, boost::noncopyable>{"PolicyInterface",

         "This class represents the base interface for policies in single agent bandits.\n"
         "\n"
         "This class represents an interface that all policies must conform to.\n"
         "The interface is generic as different methods may have very different\n"
         "ways to store and compute policies, and this interface simply asks\n"
         "for a way to sample them.\n"
         "\n"
         "In the case of bandits, the class works without requiring states.", no_init}

        .def("sampleAction",            &Bandit::PolicyInterface::sampleAction,
             "This function chooses a random action, following the policy distribution.\n"
             "\n"
             "@return The chosen action."
        , (arg("self")))

        .def("getActionProbability",    &Bandit::PolicyInterface::getActionProbability,
             "This function returns the probability of taking the specified action.\n"
             "\n"
             "@param a The selected action.\n"
             "\n"
             "@return The probability of taking the selected action.\n"
        , (arg("self"), "a"));
}
