#include <boost/python.hpp>

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

BOOST_PYTHON_MODULE(POMDP)
{
    boost::python::docstring_options localDocstringOptions(true, true, false);

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
}
