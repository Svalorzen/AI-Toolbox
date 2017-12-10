#include <boost/python.hpp>

void exportGamePolicyInterface();

void exportGameLRPPolicy();

BOOST_PYTHON_MODULE(Game)
{
    boost::python::docstring_options localDocstringOptions(true, true, false);

    exportGamePolicyInterface();

    exportGameLRPPolicy();
}
