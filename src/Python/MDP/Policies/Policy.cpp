#include <AIToolbox/MDP/Policies/Policy.hpp>

#include <boost/python.hpp>

void exportMDPPolicy() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    struct PolicyPickle : boost::python::pickle_suite {
        static boost::python::tuple getinitargs(const Policy& p) {
            return make_tuple(p.getPolicy());
        }
    };

    class_<Policy, bases<PolicyInterface>>{"Policy",

         "This class represents an MDP Policy.\n"
         "\n"
         "This class is one of the many ways to represent an MDP Policy. In\n"
         "particular, it maintains a 2 dimensional matrix of probabilities\n"
         "determining the probability of choosing an action in a given state.\n"
         "\n"
         "The class offers facilities to sample from these distributions, so\n"
         "that you can directly embed it into a decision-making process.\n"
         "\n"
         "Building this object is somewhat expensive, so it should be done\n"
         "mostly when it is known that the final solution won't change again.\n"
         "Otherwise you may want to build a wrapper around some data to\n"
         "extract the policy dynamically.", no_init}

        .def(init<size_t, size_t>(
                 "Basic constructor.\n"
                 "\n"
                 "This constructor initializes the internal policy matrix so that\n"
                 "each action in each state has the same probability of being\n"
                 "chosen (random policy). This class guarantees that at any point\n"
                 "the internal policy is a true probability distribution, i.e.\n"
                 "for each state the sum of the probabilities of choosing an action\n"
                 "sums up to 1.\n"
                 "\n"
                 "@param s The number of states of the world.\n"
                 "@param a The number of actions available to the agent."
        , (arg("self"), "s", "a")))

        .def(init<const PolicyInterface &>(
                 "Basic constructor.\n"
                 "\n"
                 "This constructor simply copies policy probability values\n"
                 "from any other compatible PolicyInterface, and stores them\n"
                 "internally. This is probably the main way you may want to use\n"
                 "this class.\n"
                 "\n"
                 "This may be a useful thing to do in case the policy that is\n"
                 "being copied is very costly to use (for example, QGreedyPolicy)\n"
                 "and it is known that it will not change anymore.\n"
                 "\n"
                 "@param p The policy which is being copied."
        , (arg("self"), "p")))

        .def(init<size_t, size_t, const ValueFunction &>(
                 "Basic constructor.\n"
                 "\n"
                 "This constructor copies the implied policy contained in a ValueFunction.\n"
                 "Keep in mind that the policy stored within a ValueFunction is\n"
                 "non-stochastic in nature, since for each state it can only\n"
                 "save a single action.\n"
                 "\n"
                 "@param s The number of states of the world.\n"
                 "@param a The number of actions available to the agent.\n"
                 "@param v The ValueFunction used as a basis for the Policy."
        , (arg("self"), "s", "a", "v")))

        .def("getPolicyMatrix",      &Policy::getPolicyMatrix, return_internal_reference<>(),
                 "This function enables inspection of the internal policy.\n"
                 "\n"
                 "@return A constant reference to the internal policy."
        , (arg("self")))

        .def_pickle(PolicyPickle());
}


