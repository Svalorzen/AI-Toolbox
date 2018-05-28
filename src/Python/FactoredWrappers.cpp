#include <boost/python.hpp>

class NamespaceFactored{};
class NamespaceFactoredMDP{};

void exportFactoredMDPJointActionLearner();

void exportFactored() {
#ifdef AITOOLBOX_EXPORT_FACTORED
    boost::python::scope x = boost::python::class_<NamespaceFactored>("Factored");

    // MDP nested scope
    {
        boost::python::scope x = boost::python::class_<NamespaceFactoredMDP>("MDP");

        exportFactoredMDPJointActionLearner();
    }
#endif
}
