#include <AIToolbox/PolicyInterface.hpp>

#include <boost/python.hpp>
#include <AIToolbox/POMDP/Types.hpp>

void exportPOMDPPolicyInterface() {
    using namespace AIToolbox;
    using namespace boost::python;

    using P = PolicyInterface<size_t, POMDP::Belief, size_t>;

    class_<P, boost::noncopyable>{"PolicyInterface",

         "This class represents the base interface for policies in POMDPs.\n"
         "\n"
         "This class represents an interface that all policies must conform to.\n"
         "The interface is generic as different methods may have very different\n"
         "ways to store and compute policies, and this interface simply asks\n"
         "for a way to sample them.\n"
         "\n"
         "In case of POMDPs, the template parameter is of type Belief, which\n"
         "allows us to sample the policy from different beliefs.", no_init}

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


