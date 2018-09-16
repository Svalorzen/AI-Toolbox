#include <AIToolbox/POMDP/Algorithms/LinearSupport.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPLinearSupport() {
    using namespace boost::python;
    using namespace AIToolbox::POMDP;

    class_<LinearSupport>{"LinearSupport",

         "This class represents the LinearSupport algorithm.\n"
         "\n"
         "This method is similar in spirit to Witness. The idea is that we look at\n"
         "certain belief points, and we try to find the best alphavectors in those\n"
         "points. Rather than looking for them though, the idea here is that we\n"
         "*know* where they are, if there are any at all.\n"
         "\n"
         "As the ValueFunction is piecewise linear and convex, if there's any\n"
         "other hyperplane that we can add to improve it, the improvements are\n"
         "going to be maximal at one of the vertices of the original surface.\n"
         "\n"
         "The idea thus is the following: first we compute the set of alphavectors\n"
         "for the corners, so we can be sure about them. Then we find all vertices\n"
         "that those alphavectors create, and we compute the error between the\n"
         "true ValueFunction and their current values.\n"
         "\n"
         "If the error is greater than a certain amount, we allow their supporting\n"
         "alphavector to join the ValueFunction, and we increase the size of the\n"
         "vertex set by adding all new vertices that are created by adding the new\n"
         "surface (and removing the ones that are made useless by it).\n"
         "\n"
         "We repeat until we have checked all available vertices, and at that\n"
         "point we are done.\n"
         "\n"
         "While this can be a very inefficient algorithm, the fact that vertices\n"
         "are checked in an orderly fashion, from highest error to lowest, allows\n"
         "if one needs it to convert this algorithm into an anytime algorithm.\n"
         "Even if there is limited time to compute the solution, the algorithm is\n"
         "guaranteed to work in the areas with high error first, allowing one to\n"
         "compute good approximations even without a lot of resources.", no_init}

        .def(init<unsigned, double>(
                 "Basic constructor.\n"
                 "\n"
                 "This constructor sets the default horizon used to solve a POMDP::Model.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the\n"
                 "constructor will throw an std::runtime_error. The tolerance\n"
                 "parameter sets the convergence criterion. A tolerance of 0.0\n"
                 "forces LinearSupport to perform a number of iterations equal to\n"
                 "the horizon specified. Otherwise, LinearSupport will stop as soon\n"
                 "as the difference between two iterations is less than the\n"
                 "tolerance specified.\n"
                 "\n"
                 "@param h The horizon chosen.\n"
                 "@param tolerance The tolerance factor to stop the value iteration loop."
        , (arg("self"), "horizon", "tolerance")))

        .def("setTolerance",                &LinearSupport::setTolerance,
                 "This function sets the tolerance parameter.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the\n"
                 "constructor will throw an std::runtime_error. The tolerance\n"
                 "parameter sets the convergence criterion. A tolerance of 0.0\n"
                 "forces LinearSupport to perform a number of iterations equal to\n"
                 "the horizon specified. Otherwise, LinearSupport will stop as soon\n"
                 "as the difference between two iterations is less than the\n"
                 "tolerance specified.\n"
                 "\n"
                 "@param t The new tolerance parameter."
        , (arg("self"), "t"))

        .def("setHorizon",                  &LinearSupport::setHorizon,
                 "This function allows setting the horizon parameter."
        , (arg("self"), "horizon"))

        .def("getTolerance",                &LinearSupport::getTolerance,
                 "This function returns the currently set tolerance parameter."
        , (arg("self")))

        .def("getHorizon",                  &LinearSupport::getHorizon,
                 "This function returns the currently set horizon parameter."
        , (arg("self")))

        .def("__call__",                    &LinearSupport::operator()<POMDPModelBinded>,
                 "This function solves a POMDP::Model completely.\n"
                 "\n"
                 "This function is pretty expensive (as are possibly all POMDP\n"
                 "solvers). It evaluates all vertices in the ValueFunction surface\n"
                 "in order to determine whether it is complete, otherwise it\n"
                 "improves it incrementally.\n"
                 "\n"
                 "@param model The POMDP model that needs to be solved.\n"
                 "\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction and the computed ValueFunction."
        , (arg("self"), "model"))

        .def("__call__",                    &LinearSupport::operator()<POMDPSparseModelBinded>,
                 "This function solves a POMDP::Model completely.\n"
                 "\n"
                 "This function is pretty expensive (as are possibly all POMDP\n"
                 "solvers). It evaluates all vertices in the ValueFunction surface\n"
                 "in order to determine whether it is complete, otherwise it\n"
                 "improves it incrementally.\n"
                 "\n"
                 "@param model The POMDP model that needs to be solved.\n"
                 "\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction and the computed ValueFunction."
        , (arg("self"), "model"));
}
