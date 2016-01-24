#include <AIToolbox/MDP/Policies/QPolicyInterface.hpp>

#include <boost/python.hpp>

void exportQPolicyInterface() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    using P = AIToolbox::PolicyInterface<size_t>;

    class_<QPolicyInterface, bases<P>, boost::noncopyable>{"QPolicyInterface", no_init};
}



