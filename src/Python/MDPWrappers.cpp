#include <boost/python.hpp>

class NamespaceMDP{};

void exportMDPTypes();

void exportMDPUtils();

void exportMDPExperience();
void exportMDPRLModel();
void exportMDPSparseExperience();
void exportMDPSparseRLModel();
void exportMDPModel();
void exportMDPSparseModel();
void exportMDPGenerativeModelPython();

void exportMDPQLearning();
void exportMDPHystereticQLearning();
void exportMDPSARSA();
void exportMDPSARSAL();
void exportMDPQL();
void exportMDPExpectedSARSA();
void exportMDPValueIteration();
void exportMDPPolicyIteration();
void exportMDPPrioritizedSweeping();
void exportMDPMCTS();

void exportMDPPolicyInterface();
void exportMDPQPolicyInterface();
void exportMDPQGreedyPolicy();
void exportMDPQSoftmaxPolicy();
void exportMDPEpsilonPolicy();
void exportMDPWoLFPolicy();
void exportMDPPolicy();

void exportMDP() {
#ifdef AITOOLBOX_EXPORT_MDP
    namespace bp = boost::python;

    // Create the module for this section
    bp::object newModule(bp::handle<>(bp::borrowed(PyImport_AddModule("AIToolbox.MDP"))));
    // Add the module to the parent's scope
    bp::scope().attr("MDP") = newModule;
    // Set the scope for the exports to the new module.
    bp::scope currentScope = newModule;

    exportMDPTypes();

    exportMDPUtils();

    exportMDPExperience();
    exportMDPSparseExperience();
    exportMDPRLModel();
    exportMDPSparseRLModel();
    exportMDPModel();
    exportMDPSparseModel();
    exportMDPGenerativeModelPython();

    exportMDPQLearning();
    exportMDPHystereticQLearning();
    exportMDPSARSA();
    exportMDPSARSAL();
    exportMDPQL();
    exportMDPExpectedSARSA();
    exportMDPValueIteration();
    exportMDPPolicyIteration();
    exportMDPPrioritizedSweeping();
    exportMDPMCTS();

    exportMDPPolicyInterface();
    exportMDPQPolicyInterface();
    exportMDPQGreedyPolicy();
    exportMDPQSoftmaxPolicy();
    exportMDPEpsilonPolicy();
    exportMDPWoLFPolicy();
    exportMDPPolicy();
#endif
}
