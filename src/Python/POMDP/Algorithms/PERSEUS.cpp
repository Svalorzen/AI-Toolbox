#include <AIToolbox/POMDP/Algorithms/PERSEUS.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPPERSEUS() {
    using namespace boost::python;
    using namespace AIToolbox::POMDP;

    class_<PERSEUS>{"PERSEUS",

         "This class implements the PERSEUS algorithm.\n"
         "\n"
         "The idea behind this algorithm is very similar to PBVI. The thing\n"
         "that changes is how beliefs are considered; in PERSEUS we only try\n"
         "to find as little VEntries as possible as to ensure that all beliefs\n"
         "considered are improved. This allows to skip generating VEntry for\n"
         "most beliefs considered, since usually few VEntry are responsible\n"
         "for supporting most of the beliefs.\n"
         "\n"
         "At the same time, this means that solutions found by PERSEUS may be\n"
         "*extremely* approximate with respect to the true Value Functions. This\n"
         "is because as long as the values for all the particle beliefs are\n"
         "increased, no matter how slightly, the algorithm stops looking - in\n"
         "effect simply guaranteeing that the worst action is never taken.\n"
         "However for many problems the solution found is actually very good,\n"
         "also given that due to the increased performance PERSEUS can do\n"
         "many more iterations than, for example, PBVI.\n"
         "\n"
         "This method is works best when it is allowed to iterate until convergence,\n"
         "and thus shouldn't be used on problems with finite horizons.", no_init}

        .def(init<size_t, unsigned, double>(
                 "Basic constructor.\n"
                 "\n"
                 "This constructor sets the default horizon/tolerance used to\n"
                 "solve a POMDP::Model and the number of beliefs used to\n"
                 "approximate the ValueFunction.\n"
                 "\n"
                 "@param nBeliefs The number of support beliefs to use.\n"
                 "@param h The horizon chosen.\n"
                 "@param tolerance The tolerance factor to stop the PERSEUS loop."
        , (arg("self"), "nBeliefs", "h", "tolerance")))

        .def("setTolerance",                &PERSEUS::setTolerance,
                 "This function sets the tolerance parameter.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the\n"
                 "constructor will throw an std::runtime_error. The tolerance\n"
                 "parameter sets the convergence criterion. A tolerance of 0.0\n"
                 "forces PERSEUS to perform a number of iterations equal to\n"
                 "the horizon specified. Otherwise, PERSEUS will stop as soon\n"
                 "as the difference between two iterations is less than the\n"
                 "tolerance specified.\n"
                 "\n"
                 "@param tolerance The new tolerance parameter."
        , (arg("self"), "tolerance"))

        .def("setHorizon",                  &PERSEUS::setHorizon,
                 "This function sets a new horizon parameter."
        , (arg("self"), "horizon"))

        .def("setBeliefSize",               &PERSEUS::setBeliefSize,
                "This function sets a new number of support beliefs."
        , (arg("self"), "nBeliefs"))

        .def("getTolerance",                &PERSEUS::getTolerance,
                 "This function returns the currently set tolerance parameter."
        , (arg("self")))

        .def("getHorizon",                  &PERSEUS::getHorizon,
                 "This function returns the currently set horizon parameter."
        , (arg("self")))

        .def("getBeliefSize",               &PERSEUS::getBeliefSize,
                 "This function returns the currently set number of support beliefs to use during a solve pass."
        , (arg("self")))

        .def("__call__",                    &PERSEUS::operator()<POMDPModelBinded>,
                 "This function solves a POMDP::Model approximately.\n"
                 "\n"
                 "This function computes a set of beliefs for which to solve\n"
                 "the input model. The beliefs are chosen stochastically,\n"
                 "trying to cover as much as possible of the belief space in\n"
                 "order to offer as precise a solution as possible.\n"
                 "\n"
                 "The final solution will try to be as small as possible, in\n"
                 "order to drastically improve performances, while at the same\n"
                 "time provide a reasonably good result.\n"
                 "\n"
                 "Note that the model input cannot have a discount of 1, due to\n"
                 "how PERSEUS initializes the value function internally; if\n"
                 "the model provided has a discount of 1 we throw.\n"
                 "\n"
                 "@param model The POMDP model that needs to be solved.\n"
                 "@param minReward The minimum reward obtainable from this model.\n"
                 "\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction and the computed ValueFunction."
        , (arg("self"), "model", "minReward"))

        .def("__call__",                    &PERSEUS::operator()<POMDPSparseModelBinded>,
                 "This function solves a POMDP::Model approximately.\n"
                 "\n"
                 "This function computes a set of beliefs for which to solve\n"
                 "the input model. The beliefs are chosen stochastically,\n"
                 "trying to cover as much as possible of the belief space in\n"
                 "order to offer as precise a solution as possible.\n"
                 "\n"
                 "The final solution will try to be as small as possible, in\n"
                 "order to drastically improve performances, while at the same\n"
                 "time provide a reasonably good result.\n"
                 "\n"
                 "Note that the model input cannot have a discount of 1, due to\n"
                 "how PERSEUS initializes the value function internally; if\n"
                 "the model provided has a discount of 1 we throw.\n"
                 "\n"
                 "@param model The POMDP model that needs to be solved.\n"
                 "@param minReward The minimum reward obtainable from this model.\n"
                 "\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction and the computed ValueFunction."
        , (arg("self"), "model", "minReward"));
}
