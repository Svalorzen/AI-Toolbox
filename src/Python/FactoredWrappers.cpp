#include <boost/python.hpp>

void exportFactoredGamePolicyInterface();

void exportFactoredGameLRPPolicy();
void exportFactoredGameESRLPolicy();

class NamespaceGame{};

BOOST_PYTHON_MODULE(Factored)
{
    boost::python::docstring_options localDocstringOptions(true, true, false);

    // Game nested scope
    {
        boost::python::scope x = boost::python::class_<NamespaceGame>("Game");
        exportFactoredGamePolicyInterface();

        exportFactoredGameLRPPolicy();
        exportFactoredGameESRLPolicy();
    }
}
