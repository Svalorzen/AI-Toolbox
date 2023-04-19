#include <boost/python.hpp>

class NamespaceMDP{};

void exportMDPTypes();

void exportMDPUtils();
void exportMDPIO();

void exportMDPExperience();
void exportMDPMaximumLikelihoodModel();
void exportMDPSparseExperience();
void exportMDPSparseMaximumLikelihoodModel();
void exportMDPModel();
void exportMDPSparseModel();
void exportMDPGenerativeModelPython();

void exportMDPQLearning();
void exportMDPRLearning();
void exportMDPDoubleQLearning();
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

void exportMDPGridWorld();
void exportMDPSimpleEnvironments();

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
    exportMDPIO();

    exportMDPExperience();
    exportMDPSparseExperience();
    exportMDPMaximumLikelihoodModel();
    exportMDPSparseMaximumLikelihoodModel();
    exportMDPModel();
    exportMDPSparseModel();
    exportMDPGenerativeModelPython();

    exportMDPQLearning();
    exportMDPRLearning();
    exportMDPDoubleQLearning();
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

    exportMDPGridWorld();
    exportMDPSimpleEnvironments();
#endif
}
