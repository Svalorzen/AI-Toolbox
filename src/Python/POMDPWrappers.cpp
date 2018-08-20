#include <boost/python.hpp>

class NamespacePOMDP{};

void exportPOMDPTypes();

void exportPOMDPUtils();

void exportPOMDPModel();
void exportPOMDPSparseModel();

void exportPOMDPPOMCP();
void exportPOMDPWitness();
void exportPOMDPIncrementalPruning();
void exportPOMDPLinearSupport();
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
    namespace bp = boost::python;

    // Create the module for this section
    bp::object newModule(bp::handle<>(bp::borrowed(PyImport_AddModule("AIToolbox.POMDP"))));
    // Add the module to the parent's scope
    bp::scope().attr("POMDP") = newModule;
    // Set the scope for the exports to the new module.
    bp::scope currentScope = newModule;

    exportPOMDPTypes();

    exportPOMDPUtils();

    exportPOMDPModel();
    exportPOMDPSparseModel();

    exportPOMDPPOMCP();
    exportPOMDPWitness();
    exportPOMDPIncrementalPruning();
    exportPOMDPLinearSupport();
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
