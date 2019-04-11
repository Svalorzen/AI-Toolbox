#include <boost/python.hpp>

class NamespaceBandit{};

void exportBanditRollingAverage();

void exportBanditPolicyInterface();

void exportBanditEpsilonPolicy();
void exportBanditRandomPolicy();
void exportBanditQGreedyPolicy();
void exportBanditQSoftmaxPolicy();
void exportBanditThompsonSamplingPolicy();

void exportBanditLRPPolicy();
void exportBanditESRLPolicy();

void exportBandit() {
#ifdef AITOOLBOX_EXPORT_BANDIT
    namespace bp = boost::python;

    // Create the module for this section
    bp::object newModule(bp::handle<>(bp::borrowed(PyImport_AddModule("AIToolbox.Bandit"))));
    // Add the module to the parent's scope
    bp::scope().attr("Bandit") = newModule;
    // Set the scope for the exports to the new module.
    bp::scope currentScope = newModule;

    exportBanditRollingAverage();

    exportBanditPolicyInterface();

    exportBanditEpsilonPolicy();
    exportBanditRandomPolicy();
    exportBanditQGreedyPolicy();
    exportBanditQSoftmaxPolicy();
    exportBanditThompsonSamplingPolicy();

    exportBanditLRPPolicy();
    exportBanditESRLPolicy();
#endif
}
