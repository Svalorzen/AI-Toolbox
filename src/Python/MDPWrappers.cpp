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

void exportMDPQLearning();
void exportMDPHystereticQLearning();
void exportMDPSARSA();
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
void exportMDPRandomPolicy();
void exportMDPPolicy();

void exportMDP() {
#ifdef AITOOLBOX_EXPORT_MDP
    boost::python::scope x = boost::python::class_<NamespaceMDP>("MDP");

    exportMDPTypes();

    exportMDPUtils();

    exportMDPExperience();
    exportMDPSparseExperience();
    exportMDPRLModel();
    exportMDPSparseRLModel();
    exportMDPModel();
    exportMDPSparseModel();

    exportMDPQLearning();
    exportMDPHystereticQLearning();
    exportMDPSARSA();
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
    exportMDPRandomPolicy();
    exportMDPPolicy();
#endif
}
