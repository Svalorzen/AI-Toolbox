#include <AIToolbox/MDP/Policies/QPolicyInterface.hpp>

#include <boost/python.hpp>

void exportMDPQPolicyInterface() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<QPolicyInterface, bases<PolicyInterface>, boost::noncopyable>{"QPolicyInterface",
         "This class is an interface to specify a policy through a QFunction.\n"
         "\n"
         "This class provides a way to sample actions without the\n"
         "need to compute a full Policy from a QFunction. This is useful\n"
         "because often many methods need to modify small parts of a Qfunction\n"
         "for progressive improvement, and computing a full Policy at each\n"
         "step can become too expensive to do.\n"
         "\n"
         "The type of policy obtained from such sampling is left to the implementation,\n"
         "since there are many ways in which such a policy may be formed.", no_init};
}



