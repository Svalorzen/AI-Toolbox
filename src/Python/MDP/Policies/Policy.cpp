#include <AIToolbox/MDP/Policies/Policy.hpp>

#include <boost/python.hpp>

void exportPolicy() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<Policy, bases<AIToolbox::PolicyInterface<size_t>>>{"Policy", init<const AIToolbox::PolicyInterface<size_t> &>()}
        .def(init<size_t, size_t, const ValueFunction &>())
        .def("getPolicyTable",      &Policy::getPolicyTable, return_internal_reference<>());
}


