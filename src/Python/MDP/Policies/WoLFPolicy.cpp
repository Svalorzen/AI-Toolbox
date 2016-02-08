#include <AIToolbox/MDP/Policies/WoLFPolicy.hpp>

#include <boost/python.hpp>

void exportWoLFPolicy() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<WoLFPolicy, bases<QPolicyInterface>>{"WoLFPolicy", init<const QFunction &, optional<double,double,double>>()}
        .def("updatePolicy",    &WoLFPolicy::updatePolicy)
        .def("setDeltaW",       &WoLFPolicy::setDeltaW)
        .def("getDeltaW",       &WoLFPolicy::getDeltaW)
        .def("setDeltaL",       &WoLFPolicy::setDeltaL)
        .def("getDeltaL",       &WoLFPolicy::getDeltaL)
        .def("setScaling",      &WoLFPolicy::setScaling)
        .def("getScaling",      &WoLFPolicy::getScaling);
}

