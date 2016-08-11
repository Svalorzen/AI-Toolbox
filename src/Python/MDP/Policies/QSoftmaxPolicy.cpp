#include <AIToolbox/MDP/Policies/QSoftmaxPolicy.hpp>

#include <boost/python.hpp>

void exportMDPQSoftmaxPolicy() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<QSoftmaxPolicy, bases<QPolicyInterface>>{"QSoftmaxPolicy",

         "This class models a softmax policy through a QFunction.\n"
         "\n"
         "A softmax policy is a policy that selects actions based on their\n"
         "expected reward: the more advantageous an action seems to be, the more\n"
         "probable its selection is. There are many ways to implement a softmax\n"
         "policy, this class implements selection using the most common method of\n"
         "sampling from a Boltzmann distribution.\n"
         "\n"
         "As the epsilon-policy, this type of policy is useful to force the agent\n"
         "to explore an unknown model, in order to gain new information to refine\n"
         "it and thus gain more reward.", no_init}

        .def(init<const QFunction &, optional<double>>(
                 "Basic constructor.\n"
                 "\n"
                 "The temperature parameter must be >= 0.0\n"
                 "otherwise the constructor will throw an std::invalid_argument.\n"
                 "\n"
                 "@param q The QFunction this policy is linked with.\n"
                 "@param temperature The parameter that controls the amount of exploration."
        , (arg("self"), "q", "temperature")))

        .def("sampleAction",        &QSoftmaxPolicy::sampleAction,
                 "This function chooses an action for state s with probability dependent on value.\n"
                 "\n"
                 "This class implements softmax through the Boltzmann\n"
                 "distribution. Thus an action will be chosen with\n"
                 "probability:\n"
                 "\n"
                 "\n"
                 "     P(a) = (e^(Q(s,a)/t))/(Sum_b{ e^(Q(s,b)/t) })\n"
                 "\n"
                 "\n"
                 "where t is the temperature. This value is not cached anywhere, so\n"
                 "continuous sampling may not be extremely fast.\n"
                 "\n"
                 "@param s The sampled state of the policy.\n"
                 "\n"
                 "@return The chosen action.\n"
        , (arg("self"), "s"))

        .def("setTemperature",      &QSoftmaxPolicy::setTemperature,
                 "This function sets the temperature parameter.\n"
                 "\n"
                 "The temperature parameter determines the amount of\n"
                 "exploration this policy will enforce when selecting actions.\n"
                 "Following the Boltzmann distribution, as the temperature\n"
                 "approaches infinity all actions will become equally\n"
                 "probable. On the opposite side, as the temperature\n"
                 "approaches zero, action selection will become completely\n"
                 "greedy.\n"
                 "\n"
                 "The temperature parameter must be >= 0.0 otherwise the\n"
                 "function will do throw std::invalid_argument.\n"
                 "\n"
                 "@param t The new temperature parameter."
        , (arg("self"), "t"))

        .def("getTemperature",      &QSoftmaxPolicy::getTemperature,
                 "This function will return the currently set temperature parameter."
        , (arg("self")));
}
