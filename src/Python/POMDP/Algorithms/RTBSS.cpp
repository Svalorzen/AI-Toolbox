#include <AIToolbox/POMDP/Algorithms/RTBSS.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

template <typename M>
void exportRTBSSByModel(std::string className) {
    using namespace AIToolbox::POMDP;
    using namespace boost::python;

    using V = RTBSS<M>;

    class_<V>{("RTBSS" + className).c_str(), (

         "This class represents the RTBSS online planner for " + className + ".\n"
         "\n"
         "This algorithm is an online planner for POMDPs. It works by pretty\n"
         "much solving the whole POMDP in a straightforward manner, but just\n"
         "for the belief it is currently in, and the horizon specified.\n"
         "\n"
         "Additionally, it uses an heuristic function in order to prune\n"
         "branches which cannot possibly help in determining which action is\n"
         "the actual best. Currently this heuristic is very crude, as it\n"
         "requires the user to manually input a maximum possible reward, and\n"
         "using it as an upper bound.\n"
         "\n"
         "Additionally, in theory one would want to explore branches from the\n"
         "most promising to the least promising, to maximize pruning. This is\n"
         "currently not done here, since an heuristic is intrinsically\n"
         "determined by a particular problem. At the same time, it is easy to\n"
         "add one, as the code specifies where one should be inserted.\n"
         "\n"
         "This method is able to return not only the best available action,\n"
         "but also the (in theory) true value of that action in the current\n"
         "belief. Note that values computed in different methods may differ\n"
         "due to floating point approximation errors.").c_str(), no_init}

        .def(init<const M&, double>(
                 "Basic constructor.\n"
                 "\n"
                 "@param m The POMDP model that POMCP will operate upon.\n"
                 "@param maxR The max reward obtainable in the model.\n"
                 "       This is used for the pruning heuristic."
        , (arg("self"), "m", "maxR")))

        .def("sampleAction",            &V::sampleAction,
                 "This function computes the best value for a given belief and its value.\n"
                 "\n"
                 "@param b The initial belief for the environment.\n"
                 "@param horizon The horizon to plan for.\n"
                 "\n"
                 "@return The best action and its value in the model."
        , (arg("self"), "b", "horizon"))

        .def("getModel",                &V::getModel,   return_value_policy<reference_existing_object>(),
                 "This function returns the POMDP generative model being used."
        , (arg("self")));
}

void exportPOMDPRTBSS() {
    using namespace AIToolbox::MDP;

    exportRTBSSByModel<POMDPModelBinded>("Model");
    exportRTBSSByModel<POMDPSparseModelBinded>("SparseModel");
}
