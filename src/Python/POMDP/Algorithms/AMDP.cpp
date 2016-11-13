#include <AIToolbox/POMDP/Algorithms/AMDP.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPAMDP() {
    using namespace boost::python;
    using namespace AIToolbox::POMDP;

    class_<AMDP>{"AMDP",

         "This class implements the Augmented MDP algorithm.\n"
         "\n"
         "This algorithm transforms a POMDP into an approximately equivalent\n"
         "MDP. This is done by extending the original POMDP statespace with\n"
         "a discretized entropy component, which approximates a sufficient\n"
         "statistic for the belief. In essence, AMDP builds states which\n"
         "contain intrinsically information about the uncertainty of the agent.\n"
         "\n"
         "In order to compute a new transition and reward function, AMDP needs\n"
         "to sample possible transitions at random, since each belief can\n"
         "potentially update to any other belief. We sample beliefs using\n"
         "the BeliefGenerator class which creates both random beliefs and\n"
         "beliefs generated using the original POMDP model, in order to try\n"
         "to obtain beliefs distributed in a way that better resembles the\n"
         "original problem.\n"
         "\n"
         "Once this is done, it is simply a matter of taking each belief,\n"
         "computing every possible new belief given an action and observation,\n"
         "and sum up all possibilities.\n"
         "\n"
         "This class also bundles together with the resulting MDP a function\n"
         "to convert an original POMDP belief into an equivalent AMDP state;\n"
         "this is done so that a policy can be applied, observation gathered\n"
         "and beliefs updated while continuing to use the approximated model.", no_init}

        .def(init<size_t, size_t>(
                 "Basic constructor.\n"
                 "\n"
                 "@param nBeliefs The number of beliefs to sample from when\n"
                 "       building the MDP model.\n"
                 "@param entropyBuckets The number of buckets into which\n"
                 "       discretize entropy."
        , (arg("self"), "nBeliefs", "entropyBuckets")))

        .def("setBeliefSize",               &AMDP::setBeliefSize,
                 "This function sets a new number of sampled beliefs."
        , (arg("self"), "nBeliefs"))

        .def("setEntropyBuckets",           &AMDP::setEntropyBuckets,
                 "This function sets the new number of buckets in which to discretize the entropy."
        , (arg("self"), "buckets"))

        .def("getBeliefSize",               &AMDP::getBeliefSize,
                 "This function returns the currently set number of sampled beliefs."
        , (arg("self")))

        .def("getEntropyBuckets",           &AMDP::getEntropyBuckets,
                 "This function returns the currently set number of entropy buckets."
        , (arg("self")))

        .def("discretizeDense",             &AMDP::discretizeDense<POMDPModelBinded>,
                 "This function constructs an approximate *dense* MDP of the provided POMDP model.\n"
                 "\n"
                 "@param model The POMDP model to be approximated.\n"
                 "\n"
                 "@return A tuple containing a dense MDP model which approximates\n"
                 "        the POMDP argument, and a function that converts a POMDP\n"
                 "        belief into a state of the MDP model."
        , (arg("self"), "model"))

        .def("discretizeSparse",            &AMDP::discretizeSparse<POMDPSparseModelBinded>,
                 "This function constructs an approximate *sparse* MDP of the provided POMDP model.\n"
                 "\n"
                 "@param model The POMDP model to be approximated.\n"
                 "\n"
                 "@return A tuple containing a sparse MDP model which approximates\n"
                 "        the POMDP argument, and a function that converts a POMDP\n"
                 "        belief into a state of the MDP model."
        , (arg("self"), "model"));
}
