#include <AIToolbox/MDP/Policies/EpsilonPolicy.hpp>

#include <boost/python.hpp>

void exportEpsilonPolicy() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<EpsilonPolicy, bases<AIToolbox::PolicyInterface<size_t>>>{"EpsilonPolicy", init<const AIToolbox::PolicyInterface<size_t> &, optional<double>>()}
        .def("setEpsilon",              &EpsilonPolicy::setEpsilon)
        .def("getEpsilon",              &EpsilonPolicy::getEpsilon);
}


