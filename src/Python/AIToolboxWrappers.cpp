#include <boost/python.hpp>

void exportTypes();
void exportBandit();
void exportMDP();
void exportPOMDP();
void exportFactored();

BOOST_PYTHON_MODULE(AIToolbox)
{
    boost::python::docstring_options localDocstringOptions(true, true, false);

    // We specify that we are creating a package This is needed since we will
    // later need to create the submodules for each category.
    boost::python::object package = boost::python::scope();
    package.attr("__path__") = "AIToolbox";

    exportTypes();
    exportBandit();
    exportMDP();
    exportPOMDP();
    exportFactored();
}
