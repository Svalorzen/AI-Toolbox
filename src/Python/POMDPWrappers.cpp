#include <boost/python.hpp>

void exportPOMDPModel();
void exportPOMDPSparseModel();

BOOST_PYTHON_MODULE(POMDP)
{
    boost::python::docstring_options localDocstringOptions(true, true, false);

    exportPOMDPModel();
    exportPOMDPSparseModel();
}
