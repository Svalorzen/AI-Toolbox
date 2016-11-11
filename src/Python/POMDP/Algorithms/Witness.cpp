#include <AIToolbox/POMDP/Algorithms/Witness.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPWitness() {
    using namespace boost::python;
    using namespace AIToolbox::POMDP;

    class_<Witness>{"Witness",

         "This class implements the Witness algorithm.\n"
         "\n"
         "This algorithm solves a POMDP Model perfectly. It computes solutions\n"
         "for each horizon incrementally, every new solution building upon the\n"
         "previous one.\n"
         "\n"
         "The Witness algorithm tries to avoid creating all possible cross-sums\n"
         "of the projected vectors. Instead, it relies on a proof that states\n"
         "that if a VEntry is suboptimal, then we can at least find a better one\n"
         "by modifying a single subtree.\n"
         "\n"
         "Given this, the Witness algorithm starts off by finding a single optimal\n"
         "VEntry for a random belief. Then, using the theorem, it knows that if a\n"
         "better VEntry exists, then there must be at least one VEntry completely\n"
         "equal to the one we just found but for a subtree, and that one will\n"
         "be better. Thus, it adds to an agenda all possible variations of the\n"
         "found optimal VEntry.\n"
         "\n"
         "From there, it examines each one of them, trying to look for a witness\n"
         "point. Once found, again it produces an optimal VEntry for that point\n"
         "and adds to the agenda all of its possible variations. VEntry which do\n"
         "not have any witness points are removed from the agenda.\n"
         "\n"
         "In addition, Witness will not add to the agenda any VEntry which it has\n"
         "already added; it uses a set to keep track of which combinations of\n"
         "subtrees it has already tried.", no_init}

        .def(init<unsigned, double>(
                 "Basic constructor.\n"
                 "\n"
                 "This constructor sets the default horizon used to solve a POMDP::Model.\n"
                 "\n"
                 "The epsilon parameter must be >= 0.0, otherwise the\n"
                 "constructor will throw an std::runtime_error. The epsilon\n"
                 "parameter sets the convergence criterion. An epsilon of 0.0\n"
                 "forces Witness to perform a number of iterations equal to\n"
                 "the horizon specified. Otherwise, Witness will stop as soon\n"
                 "as the difference between two iterations is less than the\n"
                 "epsilon specified.\n"
                 "\n"
                 "@param h The horizon chosen.\n"
                 "@param epsilon The epsilon factor to stop the value iteration loop."
        , (arg("self"), "horizon", "epsilon")))

        .def("setEpsilon",                  &Witness::setEpsilon,
                 "This function sets the epsilon parameter.\n"
                 "\n"
                 "The epsilon parameter must be >= 0.0, otherwise the\n"
                 "constructor will throw an std::runtime_error. The epsilon\n"
                 "parameter sets the convergence criterion. An epsilon of 0.0\n"
                 "forces IncrementalPruning to perform a number of iterations\n"
                 "equal to the horizon specified. Otherwise, IncrementalPruning\n"
                 "will stop as soon as the difference between two iterations\n"
                 "is less than the epsilon specified.\n"
                 "\n"
                 "@param e The new epsilon parameter."
        , (arg("self"), "e"))

        .def("setHorizon",                  &Witness::setHorizon,
                 "This function allows setting the horizon parameter."
        , (arg("self"), "horizon"))

        .def("getEpsilon",                  &Witness::getEpsilon,
                 "This function returns the currently set epsilon parameter."
        , (arg("self")))

        .def("getHorizon",                  &Witness::getHorizon,
                 "This function returns the currently set horizon parameter."
        , (arg("self")))

        .def("__call__",                    &Witness::operator()<POMDPModelBinded>,
                 "This function solves a POMDP::Model completely.\n"
                 "\n"
                 "This function is pretty expensive (as are possibly all POMDP\n"
                 "solvers).  It generates for each new solved timestep the\n"
                 "whole set of possible ValueFunctions, and prunes it\n"
                 "incrementally, trying to reduce as much as possible the\n"
                 "linear programming solves required.\n"
                 "\n"
                 "This function returns a tuple to be consistent with MDP\n"
                 "solving methods, but it should always succeed.\n"
                 "\n"
                 "@tparam M The type of POMDP model that needs to be solved.\n"
                 "\n"
                 "@param model The POMDP model that needs to be solved.\n"
                 "\n"
                 "@return A tuple containing a boolean value specifying whether\n"
                 "the specified epsilon bound was reached and the computed\n"
                 "ValueFunction."
        , (arg("self"), "model"))

        .def("__call__",                    &Witness::operator()<POMDPSparseModelBinded>,
                 "This function solves a POMDP::Model completely.\n"
                 "\n"
                 "This function is pretty expensive (as are possibly all POMDP\n"
                 "solvers).  It generates for each new solved timestep the\n"
                 "whole set of possible ValueFunctions, and prunes it\n"
                 "incrementally, trying to reduce as much as possible the\n"
                 "linear programming solves required.\n"
                 "\n"
                 "This function returns a tuple to be consistent with MDP\n"
                 "solving methods, but it should always succeed.\n"
                 "\n"
                 "@tparam M The type of POMDP model that needs to be solved.\n"
                 "\n"
                 "@param model The POMDP model that needs to be solved.\n"
                 "\n"
                 "@return A tuple containing a boolean value specifying whether\n"
                 "the specified epsilon bound was reached and the computed\n"
                 "ValueFunction."
        , (arg("self"), "model"));
}
