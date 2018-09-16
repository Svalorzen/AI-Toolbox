#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPIncrementalPruning() {
    using namespace boost::python;
    using namespace AIToolbox::POMDP;

    class_<IncrementalPruning>{"IncrementalPruning",

         "This class implements the Incremental Pruning algorithm.\n"
         "\n"
         "This algorithm solves a POMDP Model perfectly. It computes solutions\n"
         "for each horizon incrementally, every new solution building upon the\n"
         "previous one.\n"
         "\n"
         "From each solution, it computes the full set of possible\n"
         "projections. It then computes all possible cross-sums of such\n"
         "projections, in order to compute all possible vectors that can be\n"
         "included in the final solution.\n"
         "\n"
         "What makes this method unique is its pruning strategy. Instead of\n"
         "generating every possible vector, combining them and pruning, it\n"
         "tries to prune at every possible occasion in order to minimize the\n"
         "number of possible vectors at any given time. Thus it will prune\n"
         "after creating the projections, after every single cross-sum, and\n"
         "in the end when combining all projections for each action.\n"
         "\n"
         "The performances of this method are *heavily* dependent on the linear\n"
         "programming methods used. In particular, this code currently\n"
         "utilizes the lp_solve55 library. However, this library is not the\n"
         "most efficient implementation, as it defaults to a somewhat slow\n"
         "solver, and its problem-building API also tends to be slow due to\n"
         "lots of bounds checking (which are cool, but sometimes people know\n"
         "what they are doing). Still, to avoid replicating infinite amounts\n"
         "of code and managing memory by ourselves, we use its API. It would\n"
         "be nice if one day we could port directly into the code a fast lp\n"
         "implementation; for now we do what we can.", no_init}

        .def(init<unsigned, double>(
                 "Basic constructor.\n"
                 "\n"
                 "This constructor sets the default horizon used to solve a POMDP::Model.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the\n"
                 "constructor will throw an std::runtime_error. The tolerance\n"
                 "parameter sets the convergence criterion. A tolerance of 0.0\n"
                 "forces IncrementalPruning to perform a number of iterations\n"
                 "equal to the horizon specified. Otherwise, IncrementalPruning\n"
                 "will stop as soon as the difference between two iterations\n"
                 "is less than the tolerance specified.\n"
                 "\n"
                 "@param h The horizon chosen.\n"
                 "@param tolerance The tolerance factor to stop the value iteration loop."
        , (arg("self"), "horizon", "tolerance")))

        .def("setTolerance",                &IncrementalPruning::setTolerance,
                 "This function sets the tolerance parameter.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the\n"
                 "constructor will throw an std::runtime_error. The tolerance\n"
                 "parameter sets the convergence criterion. A tolerance of 0.0\n"
                 "forces IncrementalPruning to perform a number of iterations\n"
                 "equal to the horizon specified. Otherwise, IncrementalPruning\n"
                 "will stop as soon as the difference between two iterations\n"
                 "is less than the tolerance specified.\n"
                 "\n"
                 "@param t The new tolerance parameter."
        , (arg("self"), "t"))

        .def("setHorizon",                  &IncrementalPruning::setHorizon,
                 "This function allows setting the horizon parameter."
        , (arg("self"), "horizon"))

        .def("getTolerance",                &IncrementalPruning::getTolerance,
                 "This function returns the currently set tolerance parameter."
        , (arg("self")))

        .def("getHorizon",                  &IncrementalPruning::getHorizon,
                 "This function returns the currently set horizon parameter."
        , (arg("self")))

        .def("__call__",                    &IncrementalPruning::operator()<POMDPModelBinded>,
                 "This function solves a POMDP::Model completely.\n"
                 "\n"
                 "This function is pretty expensive (as are possibly all POMDP\n"
                 "solvers).  It generates for each new solved timestep the\n"
                 "whole set of possible ValueFunctions, and prunes it\n"
                 "incrementally, trying to reduce as much as possible the\n"
                 "linear programming solves required.\n"
                 "\n"
                 "@param model The POMDP model that needs to be solved.\n"
                 "\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction and the computed ValueFunction."
        , (arg("self"), "model"))

        .def("__call__",                    &IncrementalPruning::operator()<POMDPSparseModelBinded>,
                 "This function solves a POMDP::Model completely.\n"
                 "\n"
                 "This function is pretty expensive (as are possibly all POMDP\n"
                 "solvers).  It generates for each new solved timestep the\n"
                 "whole set of possible ValueFunctions, and prunes it\n"
                 "incrementally, trying to reduce as much as possible the\n"
                 "linear programming solves required.\n"
                 "\n"
                 "@param model The POMDP model that needs to be solved.\n"
                 "\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction and the computed ValueFunction."
        , (arg("self"), "model"));
}
