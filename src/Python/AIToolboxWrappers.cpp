#include <boost/python.hpp>

void exportTypes();
void exportBandit();
void exportMDP();
void exportPOMDP();
void exportFactored();

BOOST_PYTHON_MODULE(AIToolbox)
{
    boost::python::docstring_options localDocstringOptions(true, true, false);

    exportTypes();
    exportBandit();
    exportMDP();
    exportPOMDP();
    exportFactored();
}
