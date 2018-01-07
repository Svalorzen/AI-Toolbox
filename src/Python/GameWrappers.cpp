#include <boost/python.hpp>

void exportGamePolicyInterface();

void exportGameLRPPolicy();
void exportGameESRLPolicy();

BOOST_PYTHON_MODULE(Game)
{
    boost::python::docstring_options localDocstringOptions(true, true, false);

    exportGamePolicyInterface();

    exportGameLRPPolicy();
    exportGameESRLPolicy();
}
