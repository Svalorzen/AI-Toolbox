#include <AIToolbox/POMDP/Utils.hpp>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPUtils() {
    using namespace boost::python;
    using namespace AIToolbox::POMDP;

    def("updateBelief", static_cast<Belief (*)(const POMDPModelBinded &, const Belief &, const size_t, const size_t)>(updateBelief));
    def("updateBelief", static_cast<Belief (*)(const POMDPSparseModelBinded & model, const Belief &, const size_t, const size_t)>(updateBelief));
}
