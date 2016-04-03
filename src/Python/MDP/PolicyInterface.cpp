#include <AIToolbox/PolicyInterface.hpp>

#include <boost/python.hpp>

void exportPolicyInterface() {
    using namespace AIToolbox;
    using namespace boost::python;

    using P = PolicyInterface<size_t>;

    class_<P, boost::noncopyable>{"PolicyInterface",
        
         "This class represents the base interface for policies.\n"
         "\n"
         "This class represents an interface that all policies must conform to.\n"
         "The interface is generic as different methods may have very different\n"
         "ways to store and compute policies, and this interface simply asks\n"
         "for a way to sample them.\n"
         "\n"
         "This class is templatized since it works as an interface for both\n"
         "MDP and POMDP policies. In the case of MDPs, the template parameter\n"
         "State is of type size_t, which represents the states from which we are\n"
         "sampling. In case of POMDPs, the template parameter is of type Belief,\n"
         "which allows us to sample the policy from different beliefs.", no_init}

        .def("sampleAction",            &P::sampleAction,
             "This function chooses a random action for state s, following the policy distribution.\n"
             "\n"
             "@param s The sampled state of the policy.\n"
             "\n"
             "@return The chosen action."
        , (arg("self"), "s"))

        .def("getActionProbability",    &P::getActionProbability,
             "This function returns the probability of taking the specified action in the specified state.\n"
             "\n"
             "@param s The selected state.\n"
             "@param a The selected action.\n"
             "\n"
             "@return The probability of taking the selected action in the specified state.\n"
        , (arg("self"), "s", "a"));
}


