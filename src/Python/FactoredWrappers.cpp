#include <boost/python.hpp>

class NamespaceFactored{};
class NamespaceFactoredMDP{};

void exportFactoredMDPJointActionLearner();

void exportFactored() {
#ifdef AITOOLBOX_EXPORT_FACTORED
    namespace bp = boost::python;

    // Create the module for this section
    bp::object newModule(bp::handle<>(bp::borrowed(PyImport_AddModule("AIToolbox.Factored"))));
    // Add the module to the parent's scope
    bp::scope().attr("Factored") = newModule;
    // Set the scope for the exports to the new module.
    bp::scope currentScope = newModule;

    // MDP nested scope
    {
        // Create the module for this section
        bp::object newModule(bp::handle<>(bp::borrowed(PyImport_AddModule("AIToolbox.Factored.MDP"))));
        // Add the module to the parent's scope
        bp::scope().attr("MDP") = newModule;
        // Set the scope for the exports to the new module.
        bp::scope currentScope = newModule;

        exportFactoredMDPJointActionLearner();
    }
#endif
}
