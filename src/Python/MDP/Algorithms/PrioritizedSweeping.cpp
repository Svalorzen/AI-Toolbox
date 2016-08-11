#include <AIToolbox/MDP/Algorithms/PrioritizedSweeping.hpp>

#include <AIToolbox/MDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

template <typename M>
void exportPrioritizedSweepingByModel(std::string className) {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    using V = PrioritizedSweeping<M>;

    class_<V>{("PrioritizedSweeping" + className).c_str(), (

         "This class represents the PrioritizedSweeping algorithm for " + className + ".\n"
         "\n"
         "This algorithm is a refinement of the DynaQ algorithm. Instead of\n"
         "randomly sampling experienced state action pairs to get more\n"
         "information, we order each pair based on an estimate of how much\n"
         "information we can still extract from them.\n"
         "\n"
         "In particular, pairs are sorted based on the amount they modified\n"
         "the estimated ValueFunction on their last sample. This ensures that\n"
         "we always try to sample from useful pairs instead of randomly,\n"
         "extracting knowledge much faster.\n"
         "\n"
         "At the same time, this algorithm keeps a threshold for each\n"
         "state-action pair, so that it does not have to internally store all\n"
         "the pairs and save some memory/cpu time keeping the queue updated.\n"
         "Only pairs which obtained an amount of change higher than this\n"
         "treshold are kept in the queue.\n"
         "\n"
         "Differently from the QLearning and DynaQ algorithm, this class\n"
         "automatically computes the ValueFunction since it is useful to\n"
         "determine which state-action pairs are actually useful, so there's\n"
         "no need to compute it manually.\n"
         "\n"
         "Given how this algorithm updates the QFunction, the only problems\n"
         "supported by this approach are ones with an infinite horizon." ).c_str(), no_init}

        .def(init<const M&, optional<double, unsigned>>(
                 "Basic constructor.\n"
                 "\n"
                 "@param m The model to be used to update the QFunction.\n"
                 "@param theta The queue threshold.\n"
                 "@param n The number of sampling passes to do on the model upon batchUpdateQ()."
        , (arg("self"), "m", "theta", "n")))

        .def("stepUpdateQ",             &V::stepUpdateQ,
                 "This function updates the PrioritizedSweepingEigen internal update queue.\n"
                 "\n"
                 "This function updates the QFunction for the specified pair, and decides\n"
                 "whether any parent couple that can lead to this state is worth pushing\n"
                 "into the queue.\n"
                 "\n"
                 "@param s The previous state.\n"
                 "@param a The action performed."
        , (arg("self"), "s", "a"))

        .def("batchUpdateQ",            &V::batchUpdateQ,
                 "This function updates a QFunction based on simulated experience.\n"
                 "\n"
                 "In PrioritizedSweepingEigen we sample from the queue at most N times for\n"
                 "state action pairs that need updating. For each one of them we update\n"
                 "the QFunction and recursively check whether this produces new changes\n"
                 "worth updating. If so, they are inserted in the queue_ and the function\n"
                 "proceeds to the next most urgent iteration."
        , (arg("self")))

        .def("setQueueThreshold",       &V::setQueueThreshold,
                 "This function sets the theta parameter.\n"
                 "\n"
                 "The discount parameter must be >= 0.0.\n"
                 "otherwise the function will throw an std::invalid_argument.\n"
                 "\n"
                 "@param t The new theta parameter."
        , (arg("self"), "t"))

        .def("getQueueThreshold",       &V::getQueueThreshold,
                 "This function will return the currently set theta parameter."
        , (arg("self")))

        .def("setN",                    &V::setN,
                 "This function sets the number of sampling passes during batchUpdateQ()."
        , (arg("self"), "n"))

        .def("getN",                    &V::getN,
                 "This function returns the currently set number of sampling passes during batchUpdateQ()."
        , (arg("self")))

        .def("getQueueLength",          &V::getQueueLength,
                 "This function returns the current number of elements unprocessed in the queue."
        , (arg("self")))

        .def("getModel",                &V::getModel,   return_value_policy<reference_existing_object>(),
                 "This function returns a reference to the referenced Model."
        , (arg("self")))

        .def("getQFunction",            &V::getQFunction, return_internal_reference<>(),
                 "This function returns a reference to the internal QFunction."
        , (arg("self")))

        .def("setQFunction",            &V::setQFunction,
                 "This function allows you to set the value of the internal QFunction.\n"
                 "\n"
                 "This function can be useful in case you are starting with an already populated\n"
                 "Experience/Model, which you can solve (for example with ValueIteration)\n"
                 "and then improve the solution with new experience.\n"
                 "\n"
                 "@param q The QFunction that will be copied."
        , (arg("self"), "q"))

        .def("getValueFunction",        &V::getValueFunction, return_internal_reference<>(),
                 "This function returns a reference to the internal ValueFunction."
        , (arg("self")));
}

void exportMDPPrioritizedSweeping() {
    using namespace AIToolbox::MDP;

    exportPrioritizedSweepingByModel<RLModel<Experience>>("RLModel");
    exportPrioritizedSweepingByModel<SparseRLModel<SparseExperience>>("SparseRLModel");
    exportPrioritizedSweepingByModel<Model>("Model");
    exportPrioritizedSweepingByModel<SparseModel>("SparseModel");
}
