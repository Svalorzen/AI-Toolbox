#include <AIToolbox/PolicyInterface.hpp>

#include <boost/python.hpp>

void exportPolicyInterface() {
    using namespace AIToolbox;
    using namespace boost::python;

    using P = PolicyInterface<size_t>;

    class_<P, boost::noncopyable>{"PolicyInterface", no_init}
        .def("sampleAction",            &P::sampleAction)
        .def("getActionProbability",    &P::getActionProbability);
}


