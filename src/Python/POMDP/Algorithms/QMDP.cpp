#include <AIToolbox/POMDP/Algorithms/QMDP.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPQMDP() {
    using namespace AIToolbox::POMDP;
    using namespace boost::python;

    class_<QMDP>{"QMDP",

         "This class implements the QMDP algorithm.\n"
         "\n"
         "QMDP is a particular way to approach a POMDP problem and solve it\n"
         "approximately. The idea is to compute a solution that disregards the\n"
         "partial observability for all timesteps but the next one. Thus, we\n"
         "assume that after the next action the agent will suddenly be able to\n"
         "see the true state of the environment, and act accordingly. In doing\n"
         "so then, it will use an MDP value function.\n"
         "\n"
         "Remember that only the solution process acts this way. When time to\n"
         "act the QMDP solution is simply applied at every timestep, every\n"
         "time assuming that the partial observability is going to last one\n"
         "step.\n"
         "\n"
         "All in all, this class is pretty much a converter of an\n"
         "MDP::ValueFunction into a POMDP::ValueFunction.\n"
         "\n"
         "Although the solution is approximate and overconfident (since we\n"
         "assume that partial observability is going to go away, we think we\n"
         "are going to get more reward), it is still good to obtain a closer\n"
         "upper bound on the true solution. This can be used, for example, to\n"
         "boost bounds on online methods, decreasing the time they take to\n"
         "converge.\n"
         "\n"
         "The solution returned by QMDP will thus have only horizon 1, since\n"
         "the horizon requested is implicitly encoded in the MDP part of the\n"
         "solution.", no_init}

        .def(init<unsigned, double>(
                 "Basic constructor.\n"
                 "\n"
                 "QMDP uses MDP::ValueIteration in order to solve the\n"
                 "underlying MDP of the POMDP. Thus, its parameters (and\n"
                 "bounds) are the same.\n"
                 "\n"
                 "@param horizon The maximum number of iterations to perform.\n"
                 "@param tolerance The tolerance factor to stop the value iteration loop."
        , (arg("self"), "horizon", "tolerance")))

        .def("__call__",                &QMDP::operator()<POMDPModelBinded>,
                 "This function applies the QMDP algorithm on the input POMDP.\n"
                 "\n"
                 "This function computes the MDP::QFunction of the underlying MDP\n"
                 "of the input POMDP with the parameters set using ValueIteration.\n"
                 "\n"
                 "It then converts this solution into the equivalent\n"
                 "POMDP::ValueFunction. Finally it returns both (plus the\n"
                 "variation for the last iteration of ValueIteration).\n"
                 "\n"
                 "Note that no pruning is performed here, so some vectors might be\n"
                 "dominated.\n"
                 "\n"
                 "@param m The POMDP to be solved\n"
                 "\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction, the computed ValueFunction and the\n"
                 "        equivalent MDP::QFunction."
        , (arg("self"), "m"))

        .def("__call__",                &QMDP::operator()<POMDPSparseModelBinded>,
                 "This function applies the QMDP algorithm on the input POMDP.\n"
                 "\n"
                 "This function computes the MDP::QFunction of the underlying MDP\n"
                 "of the input POMDP with the parameters set using ValueIteration.\n"
                 "\n"
                 "It then converts this solution into the equivalent\n"
                 "POMDP::ValueFunction. Finally it returns both (plus the\n"
                 "variation for the last iteration of ValueIteration).\n"
                 "\n"
                 "Note that no pruning is performed here, so some vectors might be\n"
                 "dominated.\n"
                 "\n"
                 "@param m The POMDP to be solved\n"
                 "\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction, the computed ValueFunction and the\n"
                 "        equivalent MDP::QFunction."
        , (arg("self"), "m"))

        .def("setTolerance",            &QMDP::setTolerance,
                 "This function sets the tolerance parameter.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the function\n"
                 "will throw an std::invalid_argument. The tolerance parameter\n"
                 "sets the convergence criterion. A tolerance of 0.0 forces the\n"
                 "internal ValueIteration to perform a number of iterations\n"
                 "equal to the horizon specified. Otherwise, ValueIteration\n"
                 "will stop as soon as the difference between two iterations\n"
                 "is less than the tolerance specified.\n"
                 "\n"
                 "@param tolerance The new tolerance parameter."
        , (arg("self"), "tolerance"))

        .def("setHorizon",              &QMDP::setHorizon,
                 "This function sets the horizon parameter."
        , (arg("self"), "horizon"))

        .def("getTolerance",            &QMDP::getTolerance,
                 "This function returns the currently set tolerance parameter."
        , (arg("self")))

        .def("getHorizon",              &QMDP::getHorizon,
                 "This function returns the currently set horizon parameter."
        , (arg("self")));
}
