#include <AIToolbox/POMDP/Algorithms/GapMin.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPGapMin() {
    using namespace boost::python;
    using namespace AIToolbox::POMDP;

    using Retval = std::tuple<double, double, VList, AIToolbox::MDP::QFunction>;

    class_<GapMin>{"GapMin",

         "This class implements the GapMin algorithm.\n"
         "\n"
         "This method works by repeatedly refining both a lower bound and upper\n"
         "bound for the input POMDP.\n"
         "\n"
         "The lower bound is worked through PBVI.\n"
         "\n"
         "The upper bound is worked through a combination of alphavectors, and a\n"
         "belief-value pair piecewise linear surface.\n"
         "\n"
         "At each iteration, a set of beliefs are found that the algorithm thinks\n"
         "may be useful to reduce the bound.\n"
         "\n"
         "For the lower bound, these beliefs are added to a list, and run through\n"
         "PBVI. Spurious beliefs are then removed.\n"
         "\n"
         "For the upper bound, the beliefs are used to create a temporary belief\n"
         "POMDP, where each belief is a state. This belief is then used as input\n"
         "to the FastInformedBound algorithm, which refines its upper bound.\n"
         "\n"
         "The strong point of the algorithm is that beliefs are searched by gap\n"
         "size, so that the beliefs that are most likely to decrease the gap are\n"
         "looked at first. This results in less overall work to highly reduce the\n"
         "bound.\n"
         "\n"
         "In order to act, the output lower bound should be used (as it's the only\n"
         "one that gives an actual guarantee), but for this just using PBVI may be\n"
         "more useful.", no_init}

        .def(init<double, size_t>(
                 "Basic constructor.\n"
                 "\n"
                 "The input parameters can heavily influence both the time and the\n"
                 "strictness of the resulting bound.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the\n"
                 "function will throw an std::runtime_error.\n"
                 "\n"
                 "\\sa setInitialTolerance(double)\n"
                 "\\sa setPrecisionDigits(unsigned)\n"
                 "\n"
                 "@param initialTolerance The tolerance to compute the initial bounds.\n"
                 "@param precisionDigits The number of digits precision to stop the gap searching process."
        , (arg("self"), "initialTolerance", "precisionDigits")))

        .def("setInitialTolerance",                  &GapMin::setInitialTolerance,
                 "This function sets the initial tolerance used to compute the initial bounds."
                 "\n"
                 "This value is only used before having an initial bounds\n"
                 "approximation. Once that has been established, the tolerance is\n"
                 "dependent on the digits of precision parameter.\n"
                 "\n"
                 "The tolerance parameter must be >= 0.0, otherwise the\n"
                 "function will throw an std::runtime_error.\n"
                 "\n"
                 "\\sa setPrecisionDigits(unsigned);\n"
                 "\n"
                 "@param initialTolerance The new initial tolerance."
        , (arg("self"), "initialTolerance"))

        .def("getInitialTolerance",                  &GapMin::getInitialTolerance,
                 "This function returns the initial tolerance used to compute the initial bounds."
        , (arg("self")))

        .def("setPrecisionDigits",                   &GapMin::setPrecisionDigits,
                 "This function sets the digits in precision for the returned solution."
                 "\n"
                 "Depending on the values for the input model, the precision of\n"
                 "the solution is automatically adjusted to the input precision\n"
                 "digits.\n"
                 "\n"
                 "In particular, the return threshold is equal to:\n"
                 "\n"
                 "    std::pow(10, std::ceil(std::log10(std::max(std::fabs(ub), std::fabs(lb))))-precisionDigits);\n"
                 "\n"
                 "This is used in two ways:\n"
                 "\n"
                 "- To check for lower and upper bound convergence. If the bounds\n"
                 "  difference is less than the threshold, the GapMin terminates.\n"
                 "- To check for gap size converngence. If the gap has not reduced\n"
                 "  by more than the threshold during the last iteration, GapMin\n"
                 "  terminates.\n"
                 "\n"
                 "@param digits The number of digits of precision to use to test for convergence."
        , (arg("self"), "precisionDigits"))

        .def("getPrecisionDigits",                   &GapMin::getPrecisionDigits,
                 "This function returns the currently set digits of precision.\n"
                 "\n"
                 "\\sa setPrecisionDigits(unsigned);\n"
                 "\n"
                 "@return The currently set digits of precision to use to test for convergence."
        , (arg("self")))

        .def("__call__",                    static_cast<Retval(GapMin::*)(const POMDPModelBinded&, const Belief&)>(&GapMin::operator()<POMDPModelBinded>),
                 "This function efficiently computes bounds for the optimal value of the input belief for the input POMDP.\n"
                 "\n"
                 "@param model The model to compute the gap for.\n"
                 "@param initialBelief The belief to compute the gap for.\n"
                 "\n"
                 "@return The lower and upper gap bounds, the lower bound VList, and the upper bound QFunction."
        , (arg("self"), "model", "initialBelief"))

        .def("__call__",                    static_cast<Retval(GapMin::*)(const POMDPSparseModelBinded&, const Belief&)>(&GapMin::operator()<POMDPSparseModelBinded>),
                 "This function efficiently computes bounds for the optimal value of the input belief for the input POMDP.\n"
                 "\n"
                 "@param model The model to compute the gap for.\n"
                 "@param initialBelief The belief to compute the gap for.\n"
                 "\n"
                 "@return The lower and upper gap bounds, the lower bound VList, and the upper bound QFunction."
        , (arg("self"), "model", "initialBelief"));
}
