#include <boost/python.hpp>

class NamespacePOMDP{};

void exportPOMDPTypes();

void exportPOMDPUtils();

void exportPOMDPModel();
void exportPOMDPSparseModel();

void exportPOMDPPOMCP();
void exportPOMDPWitness();
void exportPOMDPIncrementalPruning();
void exportPOMDPQMDP();
void exportPOMDPRTBSS();
void exportPOMDPAMDP();
void exportPOMDPPERSEUS();
void exportPOMDPPBVI();
void exportPOMDPGapMin();

void exportPOMDPPolicyInterface();
void exportPOMDPPolicy();

void exportPOMDP() {
#ifdef AITOOLBOX_EXPORT_POMDP
    boost::python::scope x = boost::python::class_<NamespacePOMDP>("POMDP");

    exportPOMDPTypes();

    exportPOMDPUtils();

    exportPOMDPModel();
    exportPOMDPSparseModel();

    exportPOMDPPOMCP();
    exportPOMDPWitness();
    exportPOMDPIncrementalPruning();
    exportPOMDPQMDP();
    exportPOMDPRTBSS();
    exportPOMDPAMDP();
    exportPOMDPPERSEUS();
    exportPOMDPPBVI();
    exportPOMDPGapMin();

    exportPOMDPPolicyInterface();
    exportPOMDPPolicy();
#endif
}
