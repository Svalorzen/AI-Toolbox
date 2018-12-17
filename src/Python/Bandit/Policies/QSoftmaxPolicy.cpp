#include <AIToolbox/Bandit/Policies/QSoftmaxPolicy.hpp>

#include <boost/python.hpp>

void exportBanditQSoftmaxPolicy() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<QSoftmaxPolicy, bases<PolicyInterface>>{"QSoftmaxPolicy",

         "This class models a simple greedy policy.\n"
         "\n"
         "This class always selects the greediest action with respect to the\n"
         "already obtained experience.", no_init}

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
                 "     P(a) = (e^(Q(a)/t))/(Sum_b{ e^(Q(b)/t) })\n"
                 "\n"
                 "\n"
                 "where t is the temperature. This value is not cached anywhere, so\n"
                 "continuous sampling may not be extremely fast.\n"
                 "\n"
                 "@return The chosen action.\n"
        , (arg("self")))

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
