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

    def("updateBelief", static_cast<Belief (*)(const POMDPModelBinded &, const Belief &, const size_t, const size_t)>(updateBelief),
        "This function creates a new Belief by updating the input Belie\n"
        "with the input action and observation, following th\n"
        "transition/observation functions contained in the input model\n"
        "\n"
        "@param model The model used to update the belief\n"
        "@param b The old belief\n"
        "@param a The action taken during the transition\n"
        "@param o The observation registered"
    , (args("model"), "b", "a", "o")
    );
    def("updateBelief", static_cast<Belief (*)(const POMDPSparseModelBinded & model, const Belief &, const size_t, const size_t)>(updateBelief),
        "This function creates a new Belief by updating the input Belie\n"
        "with the input action and observation, following th\n"
        "transition/observation functions contained in the input model\n"
        "\n"
        "@param model The model used to update the belief\n"
        "@param b The old belief\n"
        "@param a The action taken during the transition\n"
        "@param o The observation registered"
    , (args("model"), "b", "a", "o")
    );
}
