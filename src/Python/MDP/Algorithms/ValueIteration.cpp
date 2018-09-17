#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

#include <AIToolbox/MDP/Types.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportMDPValueIteration() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<ValueIteration>{"ValueIteration",

         "This class applies the value iteration algorithm.\n"
         "\n"
         "This algorithm solves an MDP model for the specified horizon, or less\n"
         "if convergence is encountered.\n"
         "\n"
         "The idea of this algorithm is to iteratively compute the\n"
         "ValueFunction for the MDP optimal policy. On the first iteration,\n"
         "the ValueFunction for horizon 1 is obtained. On the second\n"
         "iteration, the one for horizon 2. This process is repeated until the\n"
         "ValueFunction has converged within a certain accuracy, or the\n"
         "horizon requested is reached.\n"
         "\n"
         "This implementation in particular is ported from the MATLAB\n"
         "MDPToolbox (although it is simplified).", no_init}

        .def(init<unsigned, optional<double>>(
                 "Basic constructor.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the\n"
                 "constructor will throw an std::runtime_error. The tolerance\n"
                 "parameter sets the convergence criterion. A tolerance of 0.0\n"
                 "forces ValueIteration to perform a number of iterations\n"
                 "equal to the horizon specified. Otherwise, ValueIteration\n"
                 "will stop as soon as the difference between two iterations\n"
                 "is less than the tolerance specified.\n"
                 "\n"
                 "@param horizon The maximum number of iterations to perform.\n"
                 "@param tolerance The tolerance factor to stop the value iteration loop."
        , (arg("self"), "horizon", "tolerance")))

        .def("__call__",                &ValueIteration::operator()<Model>,
                 "This function applies value iteration on an MDP to solve it.\n"
                 "\n"
                 "The algorithm is constrained by the currently set parameters.\n"
                 "\n"
                 "@param m The MDP that needs to be solved.\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction, the computed ValueFunction and the\n"
                 "        QFunction for the Model."
        , (arg("self"), "m"))

        .def("__call__",                &ValueIteration::operator()<SparseModel>,
                 "This function applies value iteration on an MDP to solve it.\n"
                 "\n"
                 "The algorithm is constrained by the currently set parameters.\n"
                 "\n"
                 "@param m The MDP that needs to be solved.\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction, the computed ValueFunction and the\n"
                 "        QFunction for the Model."
        , (arg("self"), "m"))

        .def("__call__",                &ValueIteration::operator()<RLModel<Experience>>,
                 "This function applies value iteration on an MDP to solve it.\n"
                 "\n"
                 "The algorithm is constrained by the currently set parameters.\n"
                 "\n"
                 "@param m The MDP that needs to be solved.\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction, the computed ValueFunction and the\n"
                 "        QFunction for the Model."
        , (arg("self"), "m"))

        .def("__call__",                &ValueIteration::operator()<SparseRLModel<SparseExperience>>,
                 "This function applies value iteration on an MDP to solve it.\n"
                 "\n"
                 "The algorithm is constrained by the currently set parameters.\n"
                 "\n"
                 "@param m The MDP that needs to be solved.\n"
                 "@return A tuple containing the maximum variation for the\n"
                 "        ValueFunction, the computed ValueFunction and the\n"
                 "        QFunction for the Model."
        , (arg("self"), "m"))

        .def("setTolerance",            &ValueIteration::setTolerance,
                 "This function sets the tolerance parameter.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the\n"
                 "constructor will throw an std::runtime_error. The tolerance\n"
                 "parameter sets the convergence criterion. An tolerance of 0.0\n"
                 "forces ValueIteration to perform a number of iterations\n"
                 "equal to the horizon specified. Otherwise, ValueIteration\n"
                 "will stop as soon as the difference between two iterations\n"
                 "is less than the tolerance specified.\n"
                 "\n"
                 "@param t The new tolerance parameter."
        , (arg("self"), "t"))

        .def("setHorizon",              &ValueIteration::setHorizon,
                 "This function sets the horizon parameter."
        , (arg("self"), "horizon"))

        .def("getTolerance",            &ValueIteration::getTolerance,
                 "This function will return the currently set tolerance parameter."
        , (arg("self")))

        .def("getHorizon",              &ValueIteration::getHorizon,
                 "This function will return the current horizon parameter."
        , (arg("self")));
}
