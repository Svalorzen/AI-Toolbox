#include <boost/python.hpp>

class NamespaceFactored{};
class NamespaceFactoredGame{};
class NamespaceFactoredMDP{};

void exportFactoredGamePolicyInterface();

void exportFactoredGameLRPPolicy();
void exportFactoredGameESRLPolicy();

void exportFactoredMDPJointActionLearner();

void exportFactored() {
#ifdef AITOOLBOX_EXPORT_FACTORED
    boost::python::scope x = boost::python::class_<NamespaceFactored>("Factored");

    // Game nested scope
    {
        boost::python::scope x = boost::python::class_<NamespaceFactoredGame>("Game");

        exportFactoredGamePolicyInterface();

        exportFactoredGameLRPPolicy();
        exportFactoredGameESRLPolicy();
    }

    // MDP nested scope
    {
        boost::python::scope x = boost::python::class_<NamespaceFactoredMDP>("MDP");

        exportFactoredMDPJointActionLearner();
    }
#endif
}
