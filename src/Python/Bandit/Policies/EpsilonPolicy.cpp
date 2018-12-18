#include <AIToolbox/Bandit/Policies/EpsilonPolicy.hpp>
#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>

#include <boost/python.hpp>

void exportBanditEpsilonPolicy() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<EpsilonPolicy, bases<PolicyInterface>>{"EpsilonPolicy",

         "This class is a policy wrapper for epsilon action choice.\n"
         "\n"
         "This class is used to wrap already existing policies to implement\n"
         "automatic exploratory behaviour (e.g. epsilon-greedy policies).\n"
         "\n"
         "An epsilon-greedy policy is a policy that takes a greedy action a\n"
         "certain percentage of the time, and otherwise takes a random action.\n"
         "They are useful to force the agent to explore an unknown model, in order\n"
         "to gain new information to refine it and thus gain more reward.\n"
         "\n"
         "Please note that to obtain an epsilon-greedy policy the wrapped\n"
         "policy needs to already be greedy with respect to the model.", no_init}

        .def(init<const PolicyInterface &, optional<double>>(
             "Basic constructor.\n"
             "\n"
             "This constructor saves the input policy and the epsilon\n"
             "parameter for later use.\n"
             "\n"
             "The epsilon parameter must be >= 0.0 and <= 1.0,\n"
             "otherwise the constructor will throw an std::invalid_argument.\n"
             "\n"
             "@param p The policy that is being extended.\n"
             "@param epsilon The parameter that controls the amount of exploration."
        , (arg("self"), "p", "epsilon")))

        .def("sampleAction",            &EpsilonPolicy::sampleAction,
             "This function chooses an action for state s, following the policy distribution and epsilon.\n"
             "\n"
             "This function has a probability of (1 - epsilon) of selecting\n"
             "a random action. Otherwise, it selects an action according\n"
             "to the distribution specified by the wrapped policy.\n"
             "\n"
             "@return The chosen action."
        , (arg("self")))

        .def("getActionProbability",    &EpsilonPolicy::getActionProbability,
             "This function returns the probability of taking the specified action.\n"
             "\n"
             "This function takes into account parameter epsilon\n"
             "while computing the final probability.\n"
             "\n"
             "@param a The selected action.\n"
             "\n"
             "@return The probability of taking the selected action."
        , (arg("self"), "a"))

        .def("setEpsilon",              &EpsilonPolicy::setEpsilon,
             "This function sets the epsilon parameter.\n"
             "\n"
             "The epsilon parameter determines the amount of exploration this\n"
             "policy will enforce when selecting actions. In particular\n"
             "actions are going to selected randomly with probability\n"
             "(1-epsilon), and are going to be selected following the\n"
             "underlying policy with probability epsilon.\n"
             "\n"
             "The epsilon parameter must be >= 0.0 and <= 1.0,\n"
             "otherwise the function will do throw std::invalid_argument.\n"
             "\n"
             "@param e The new epsilon parameter."
        , (arg("self"), "e"))

        .def("getEpsilon",              &EpsilonPolicy::getEpsilon,
             "This function will return the currently set epsilon parameter."
        , (arg("self")));
}
