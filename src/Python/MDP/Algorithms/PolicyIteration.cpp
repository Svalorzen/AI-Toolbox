#include <AIToolbox/MDP/Algorithms/PolicyIteration.hpp>

#include <AIToolbox/MDP/Types.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportMDPPolicyIteration() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<PolicyIteration>{"PolicyIteration",

         "This class applies the policy iteration algorithm.\n"
         "\n"
         "This algorithm begins with an arbitrary policy (random), and uses\n"
         "the PolicyEvaluation algorithm to find out the Values for each state\n"
         "of this policy.\n"
         "\n"
         "Once this is done, the policy can be improved by using a greedy\n"
         "approach towards the QFunction found. The new policy is then newly\n"
         "evaluated, and the process repeated.\n"
         "\n"
         "When the policy does not change anymore, it is guaranteed to be\n"
         "optimal, and the found QFunction is returned.\n", no_init}

        .def(init<unsigned, optional<double>>(
                "Basic constructor.\n"
                "\n"
                "@param horizon The horizon parameter to use during the PolicyEvaluation phase.\n"
                "@param tolerance The tolerance parameter to use during the PolicyEvaluation phase."
        , (arg("self"), "horizon", "tolerance")))

        .def("__call__",                &PolicyIteration::operator()<Model>,
                "This function applies policy iteration on an MDP to solve it.\n"
                "\n"
                "The algorithm is constrained by the currently set parameters.\n"
                "\n"
                "@param m The MDP that needs to be solved."
                "@return The QFunction of the optimal policy found."
        , (arg("self"), "m"))

        .def("__call__",                &PolicyIteration::operator()<SparseModel>,
                "This function applies policy iteration on an MDP to solve it.\n"
                "\n"
                "The algorithm is constrained by the currently set parameters.\n"
                "\n"
                "@param m The MDP that needs to be solved."
                "@return The QFunction of the optimal policy found."
        , (arg("self"), "m"))

        .def("__call__",                &PolicyIteration::operator()<RLModel<Experience>>,
                "This function applies policy iteration on an MDP to solve it.\n"
                "\n"
                "The algorithm is constrained by the currently set parameters.\n"
                "\n"
                "@param m The MDP that needs to be solved."
                "@return The QFunction of the optimal policy found."
        , (arg("self"), "m"))

        .def("__call__",                &PolicyIteration::operator()<SparseRLModel<SparseExperience>>,
                "This function applies policy iteration on an MDP to solve it.\n"
                "\n"
                "The algorithm is constrained by the currently set parameters.\n"
                "\n"
                "@param m The MDP that needs to be solved."
                "@return The QFunction of the optimal policy found."
        , (arg("self"), "m"))

        .def("setTolerance",            &PolicyIteration::setTolerance,
                "This function sets the tolerance parameter.\n"
                "\n"
                "The tolerance parameter must be >= 0 or the function will throw."
        , (arg("self"), "e"))

        .def("setHorizon",              &PolicyIteration::setHorizon,
                 "This function sets the horizon parameter."
        , (arg("self"), "horizon"))

        .def("getTolerance",            &PolicyIteration::getTolerance,
                 "This function will return the currently set tolerance parameter."
        , (arg("self")))

        .def("getHorizon",              &PolicyIteration::getHorizon,
                 "This function will return the current horizon parameter."
        , (arg("self")));
}
