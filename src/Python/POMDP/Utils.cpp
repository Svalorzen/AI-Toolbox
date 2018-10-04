#include <AIToolbox/POMDP/Utils.hpp>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

#include <AIToolbox/Utils/Polytope.hpp>
#include "../Utils.hpp"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

/// Wrapper for Python of the same named function
double computeOptimisticValueWrapper(const AIToolbox::Vector & p, const std::vector<std::pair<AIToolbox::Vector, double>> & pvPairs) {
    return AIToolbox::computeOptimisticValue(p, std::begin(pvPairs), std::end(pvPairs));
}
/// Wrapper for Python of the same named function
std::vector<std::pair<AIToolbox::Vector, double>> findVerticesNaiveWrapper(const std::vector<AIToolbox::Vector> & tests, const std::vector<AIToolbox::Vector> & planes) {
    return AIToolbox::findVerticesNaive(std::begin(tests), std::end(tests), std::begin(planes), std::end(planes));
}

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPUtils() {
    using namespace boost::python;
    using namespace AIToolbox::POMDP;

    def("updateBelief", static_cast<Belief (*)(const POMDPModelBinded &, const Belief &, const size_t, const size_t)>(updateBelief),
        "This function creates a new Belief by updating the input Belie\n"
        "with the input action and observation, following th\n"
        "transition/observation functions contained in the input model\n"
        "\n"
        "@param model The model used to update the belief\n"
        "@param b The old belief\n"
        "@param a The action taken during the transition\n"
        "@param o The observation registered"
    , (args("model"), "b", "a", "o")
    );
    def("updateBelief", static_cast<Belief (*)(const POMDPSparseModelBinded & model, const Belief &, const size_t, const size_t)>(updateBelief),
        "This function creates a new Belief by updating the input Belie\n"
        "with the input action and observation, following th\n"
        "transition/observation functions contained in the input model\n"
        "\n"
        "@param model The model used to update the belief\n"
        "@param b The old belief\n"
        "@param a The action taken during the transition\n"
        "@param o The observation registered"
    , (args("model"), "b", "a", "o")
    );

    // We'll move the function below in another file at some point.
    using VVPair = std::pair<AIToolbox::Vector, double>;
    PairFromPython<VVPair>();
    PairToPython<VVPair>();
    VectorFromPython<VVPair>();
    class_<std::vector<VVPair>>{"VVVector"}
        .def(vector_indexing_suite<std::vector<VVPair>, true>());
    def("computeOptimisticValue", computeOptimisticValueWrapper,
         "This function computes the optimistic value of a point given known vertices and values.\n"
         "\n"
         "This function computes an LP to determine the best possible value of a\n"
         "point given all known best vertices around it.\n"
         "\n"
         "This function is needed in multi-objective settings (rather than\n"
         "POMDPs), since the step where we compute the optimal value for a given\n"
         "point is extremely expensive (it requires solving a full MDP). Thus\n"
         "linear programming is used in order to determine an optimistic bound\n"
         "when deciding the next point to extract from the queue during the linear\n"
         "support process.\n"
         "\n"
         "@param p The point where we want to compute the best possible value.\n"
         "@param pvPairs A list of point-value pairs representing all surrounding vertices.\n"
         "\n"
         "@return The best possible value that the input point can have given the known vertices."
         , (arg("p"), "pvPairs")
    );
    def("findVerticesNaive", findVerticesNaiveWrapper,
         "This function implements a naive vertex enumeration algorithm.\n"
         "\n"
         "This function goes through every subset of planes of size S, and finds\n"
         "all vertices it can. In particular, it goes through the first list one\n"
         "element at a time, and joins it with S-1 elements from the second list.\n"
         "\n"
         "Even more precisely, we take >= 1 elements from the second list. The\n"
         "remaining elements (so that in total we still use S-1) are simply the\n"
         "simplex boundaries, which allows us to find the corners located there.\n"
         "\n"
         "This method may find duplicate vertices (it does not bother to prune\n"
         "them), as a vertex can be in the convergence of more than S planes.\n"
         "\n"
         "The advantage is that we do not need any linear programming, and simple\n"
         "matrix decomposition techniques suffice.\n"
         "\n"
         "Warning: the values of each vertex depends on the planes it has been\n"
         "found of, and thus may *not* be the true value if considering all planes\n"
         "at the same time!\n"
         "\n"
         "@param tests The range of the planes to find vertices for.\n"
         "@param planes The range of all other planes.\n"
         "\n"
         "@return A non-unique list of all the vertices found."
        , (arg("tests"), "planes")
    );
}
