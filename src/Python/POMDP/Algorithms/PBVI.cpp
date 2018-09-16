#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPPBVI() {
    using namespace boost::python;
    using namespace AIToolbox::POMDP;

    class_<PBVI>{"PBVI",

         "This class implements the Point Based Value Iteration algorithm.\n"
         "\n"
         "The idea behind this algorithm is to solve a POMDP Model\n"
         "approximately. When computing a perfect solution, the main problem\n"
         "is pruning the resulting ValueFunction in order to contain only a\n"
         "parsimonious representation. What this means is that many vectors\n"
         "inside can be dominated by others, and so they do not add any\n"
         "additional information, while at the same time occupying memory and\n"
         "computational time.\n"
         "\n"
         "The way this method tries to fix the problem is by solving the Model\n"
         "in a set of specified Beliefs. Doing so results in no need for\n"
         "pruning at all, since every belief uniquely identifies one of the\n"
         "optimal solution vectors (only uniqueness in the final set is\n"
         "required, but it is way cheaper than linear programming).\n"
         "\n"
         "The set of Beliefs are stochastically computed as to cover as much\n"
         "as possible of the belief space, to ensure minimization of the final\n"
         "error. The final solution will thus be correct 100% in the Beliefs\n"
         "that have been selected, and will (possibly) overshoot in\n"
         "non-covered Beliefs.\n"
         "\n"
         "In addition, the fact that we solve only for a fixed set of Beliefs\n"
         "guarantees that our final solution is limited in size, which is\n"
         "useful since even small POMDP true solutions can explode in size\n"
         "with high horizons, for very little gain.\n"
         "\n"
         "There is no convergence guarantee of this method, but the error is\n"
         "bounded.", no_init}

        .def(init<size_t, unsigned, double>(
                 "Basic constructor.\n"
                 "\n"
                 "This constructor sets the default horizon/tolerance used to\n"
                 "solve a POMDP::Model and the number of beliefs used to\n"
                 "approximate the ValueFunction.\n"
                 "\n"
                 "@param nBeliefs The number of support beliefs to use.\n"
                 "@param h The horizon chosen.\n"
                 "@param tolerance The tolerance factor to stop the PBVI loop."
        , (arg("self"), "nBeliefs", "h", "tolerance")))

        .def("setTolerance",                &PBVI::setTolerance,
                 "This function sets the tolerance parameter.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the\n"
                 "constructor will throw an std::runtime_error. The tolerance\n"
                 "parameter sets the convergence criterion. A tolerance of 0.0\n"
                 "forces PBVI to perform a number of iterations equal to\n"
                 "the horizon specified. Otherwise, PBVI will stop as soon\n"
                 "as the difference between two iterations is less than the\n"
                 "tolerance specified.\n"
                 "\n"
                 "@param t The new tolerance parameter."
        , (arg("self"), "t"))

        .def("setHorizon",                  &PBVI::setHorizon,
                 "This function sets a new horizon parameter."
        , (arg("self"), "horizon"))

        .def("setBeliefSize",               &PBVI::setBeliefSize,
                "This function sets a new number of support beliefs."
        , (arg("self"), "nBeliefs"))

        .def("getTolerance",                &PBVI::getTolerance,
                 "This function returns the currently set tolerance parameter."
        , (arg("self")))

        .def("getHorizon",                  &PBVI::getHorizon,
                 "This function returns the currently set horizon parameter."
        , (arg("self")))

        .def("getBeliefSize",               &PBVI::getBeliefSize,
                 "This function returns the currently set number of support beliefs to use during a solve pass."
        , (arg("self")))

        .def("__call__",                    static_cast<std::tuple<double, ValueFunction>(PBVI::*)(const POMDPModelBinded&, ValueFunction)>(&PBVI::operator()<POMDPModelBinded>),
                 "This function solves a POMDP::Model approximately.\n"
                 "\n"
                 "This function computes a set of beliefs for which to solve\n"
                 "the input model. The beliefs are chosen stochastically,\n"
                 "trying to cover as much as possible of the belief space in\n"
                 "order to offer as precise a solution as possible. The final\n"
                 "solution will only contain ValueFunctions for those Beliefs\n"
                 "and will interpolate them for points it did not solve for.\n"
                 "Even though the resulting solution is approximate very often\n"
                 "it is good enough, and this comes with an incredible\n"
                 "increase in speed.\n"
                 "\n"
                 "Note that even in the beliefs sampled the solution is not\n"
                 "guaranteed to be optimal. This is because a solution for\n"
                 "horizon h can only be computed with the true solution from\n"
                 "horizon h-1. If such a solution is approximate (and it is\n"
                 "here), then the solution for h will not be optimal by\n"
                 "definition.\n"
                 "\n"
                 "@param model The POMDP model that needs to be solved.\n"
                 "@param v The ValueFunction to startup the process from, if needed.\n"
                 "\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction and the computed ValueFunction."
        , (arg("self"), "model", "v"))

        .def("__call__",                    static_cast<std::tuple<double, ValueFunction>(PBVI::*)(const POMDPSparseModelBinded&, ValueFunction)>(&PBVI::operator()<POMDPSparseModelBinded>),
                 "This function solves a POMDP::Model approximately.\n"
                 "\n"
                 "This function computes a set of beliefs for which to solve\n"
                 "the input model. The beliefs are chosen stochastically,\n"
                 "trying to cover as much as possible of the belief space in\n"
                 "order to offer as precise a solution as possible. The final\n"
                 "solution will only contain ValueFunctions for those Beliefs\n"
                 "and will interpolate them for points it did not solve for.\n"
                 "Even though the resulting solution is approximate very often\n"
                 "it is good enough, and this comes with an incredible\n"
                 "increase in speed.\n"
                 "\n"
                 "Note that even in the beliefs sampled the solution is not\n"
                 "guaranteed to be optimal. This is because a solution for\n"
                 "horizon h can only be computed with the true solution from\n"
                 "horizon h-1. If such a solution is approximate (and it is\n"
                 "here), then the solution for h will not be optimal by\n"
                 "definition.\n"
                 "\n"
                 "@param model The POMDP model that needs to be solved.\n"
                 "@param v The ValueFunction to startup the process from, if needed.\n"
                 "\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction and the computed ValueFunction."
        , (arg("self"), "model", "v"));
}
