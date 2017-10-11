#include <AIToolbox/MDP/Utils.hpp>

#include <boost/python.hpp>

void exportMDPUtils() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    def("makeQFunction", makeQFunction,
        "This function creates an AIToolbox QFunction structure. This can be\n"
        "directly useful when creating multiple interdependent AIToolbox classes\n"
        "which also need to share a single QFunction (e.g. ExpectedSARSA and\n"
        "QGreedyPolicy).\n"
        "\n"
        "@param S The size of the state space of the QFunction.\n"
        "@param A The size of the action space of the QFunction."
        , (arg("S"), "A")
    );
}
