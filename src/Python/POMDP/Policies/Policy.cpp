#include <AIToolbox/POMDP/Policies/Policy.hpp>

#include <boost/python.hpp>

void exportPOMDPPolicy() {
    using namespace AIToolbox::POMDP;
    using namespace boost::python;

    class_<Policy, bases<Policy::Base>>{"Policy",

         "This class represents a POMDP Policy.\n"
         "\n"
         "This class currently represents a basic Policy adaptor for a\n"
         "POMDP::ValueFunction. What this class does is to extract the policy\n"
         "tree contained within a POMDP::ValueFunction. The idea is that, at\n"
         "each horizon, the ValueFunction contains a set of applicable\n"
         "solutions (alpha vectors) for the POMDP. At each Belief point, only\n"
         "one of those vectors applies.\n"
         "\n"
         "This class finds out at every belief which is the vector that\n"
         "applies, and returns the appropriate action. At the same time, it\n"
         "provides facilities to follow the chosen vector along the tree\n"
         "(since future actions depend on the observations obtained by the\n"
         "agent).", no_init}

    .def(init<size_t, size_t, size_t>(
                 "Basic constrctor.\n"
                 "\n"
                 "This constructor initializes the internal ValueFunction as\n"
                 "having only the horizon 0 no values solution. This is most\n"
                 "useful if the Policy needs to be read from a file.\n"
                 "\n"
                 "@param s The number of states of the world.\n"
                 "@param a The number of actions available to the agent.\n"
                 "@param o The number of possible observations the agent could make."
    , (arg("self"), "s", "a", "o")))

    .def(init<size_t, size_t, size_t, const ValueFunction&>(
                 "Basic constrctor.\n"
                 "\n"
                 "This constructor copies the implied policy contained in a\n"
                 "ValueFunction.  Keep in mind that the policy stored within a\n"
                 "ValueFunction is non-stochastic in nature, since for each\n"
                 "state it can only save a single action.\n"
                 "\n"
                 "@param s The number of states of the world.\n"
                 "@param a The number of actions available to the agent.\n"
                 "@param o The number of possible observations the agent could make.\n"
                 "@param v The ValueFunction used as a basis for the Policy."
    , (arg("self"), "s", "a", "o", "v")))

    .def("sampleAction",    static_cast<std::tuple<size_t,size_t>(Policy::*)(const Belief&,unsigned) const>(&Policy::sampleAction),
                 "This function chooses a random action for belief b when horizon steps are missing, following the policy distribution.\n"
                 "\n"
                 "There are a couple of differences between this sampling\n"
                 "function and the simpler version. The first one is that this\n"
                 "function is actually able to sample from different\n"
                 "timesteps, since this class is able to maintain a full\n"
                 "policy tree over time.\n"
                 "\n"
                 "The second difference is that it returns two values. The\n"
                 "first one is the requested action.  The second return value\n"
                 "is an id that allows the policy to compute more efficiently\n"
                 "the sampled action during the next timestep, if provided to\n"
                 "the Policy together with the obtained observation.\n"
                 "\n"
                 "@param b The sampled belief of the policy.\n"
                 "@param horizon The requested horizon, meaning the number of\n"
                 "               timesteps missing until the end of the\n"
                 "               'episode'. horizon 0 will return a valid, \n"
                 "               non-specified action.\n"
                 "\n"
                 "@return A tuple containing the chosen action, plus an id\n"
                 "        useful to sample an action more efficiently at the\n"
                 "        next timestep, if required."
    , (arg("self"), "b", "horizon"))

    .def("sampleAction",    static_cast<std::tuple<size_t,size_t>(Policy::*)(size_t,size_t,unsigned) const>(&Policy::sampleAction),
                 "This function chooses a random action after performing a sampled action and observing observation o, for a particular horizon.\n"
                 "\n"
                 "This sampling function is provided in case an already\n"
                 "sampled action has been performed, an observation\n"
                 "registered, and now a new action is needed for the next\n"
                 "timestep. Using this function is highly recommended, as no\n"
                 "belief update is necessary, and no lookup in a possibly very\n"
                 "long list of VEntries required.\n"
                 "\n"
                 "Note that this function works if and only if the horizon is\n"
                 "going to be 1 (one) less than the value used for the\n"
                 "previous sampling, otherwise anything could happen.\n"
                 "\n"
                 "To keep things simple, the id does not store internally the\n"
                 "needed horizon value, and you are requested to keep track of\n"
                 "it yourself.\n"
                 "\n"
                 "An example of usage for this function would be:\n"
                 "\n"
                 "~~~~~~~~~~~~~~~~~~~~~~~{.cpp}\n"
                 "horizon = 3;\n"
                 "// First sample\n"
                 "auto result = sampleAction(belief, horizon);\n"
                 "// We do the action, something happens, we get an observation.\n"
                 "size_t observation = performAction(std::get<0>(result));\n"
                 "--horizon;\n"
                 "// We sample again, after reducing the horizon, with the previous id.\n"
                 "result = sampleAction(std::get<1>(result), observation, horizon);\n"
                 "~~~~~~~~~~~~~~~~~~~~~~~\n"
                 "\n"
                 "@param id An id returned from a previous call of sampleAction.\n"
                 "@param o The observation obtained after performing a previously\n"
                 "         sampled action.\n"
                 "@param horizon The new horizon, equal to the old sampled horizon-1.\n"
                 "\n"
                 "@return A tuple containing the chosen action, plus an id\n"
                 "        useful to sample an action more efficiently at the\n"
                 "        next timestep, if required."
    , (arg("self"), "id", "o", "horizon"))

    .def("getActionProbability", static_cast<double(Policy::*)(const Belief&,size_t,unsigned) const>(&Policy::getActionProbability),
                 "This function returns the probability of taking the specified action in the specified belief.\n"
                 "\n"
                 "@param b The selected belief.\n"
                 "@param a The selected action.\n"
                 "@param horizon The requested horizon, meaning the number of timesteps missing until\n"
                 "the end of the 'episode'.\n"
                 "\n"
                 "@return The probability of taking the selected action in\n"
                 "        the specified belief in the specified horizon."
    , (arg("self"), "b", "a", "horizon"))

    .def("getO",                        &Policy::getO,
                 "This function returns the number of observations possible for the agent."
    , (arg("self")))

    .def("getH",                        &Policy::getH,
                 "This function returns the highest horizon available within this Policy.\n"
                 "\n"
                 "Note that all functions that accept an horizon as a\n"
                 "parameter DO NOT check the bounds of that variable. In\n"
                 "addition, note that while for S,A,O getters you get a number\n"
                 "that exceeds by 1 the values allowed (since counting starts\n"
                 "from 0), here the bound is actually included in the limit,\n"
                 "as horizon 0 does not really do anything.\n"
                 "\n"
                 "Example: getH() returns 5. This means that 5 is the highest\n"
                 "allowed parameter for an horizon in any other Policy method.\n"
                 "\n"
                 "@return The highest horizon policied."
    , (arg("self")))

    .def("getValueFunction",            &Policy::getValueFunction, return_internal_reference<>(),
                 "This function returns the internally stored ValueFunction."
    , (arg("self")));
}

