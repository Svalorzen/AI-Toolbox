#include <boost/python.hpp>
#include "GenerativeModelPython.hpp"

void exportMDPGenerativeModelPython() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<GenerativeModelPython>{"GenerativeModelPython",

         "This class allows to import generative models from Python."
         "\n"
         "This class wraps an instance of a Python class that provides generator\n"
         "methods to sample states and rewards from, so that one does not need to\n"
         "always specify transition and reward functions from Python.", no_init}

        .def(init<boost::python::object>(
                 "Basic constructor."
                 "\n"
                 "This constructor takes a Python object, which will be used to\n"
                 "call the generative methods from C++.\n"
                 "\n"
                 "This class expects the instance to have at least the following methods:\n"
                 "\n"
                 "- getS(): returns the number of states of the environment.\n"
                 "- getA(): returns the number of actions of the environment, in ALL states.\n"
                 "- getDiscount(): returns the discount of the environment, [0, 1].\n"
                 "- isTerminal(s): returns whether a given state is a terminal state.\n"
                 "- sampleSR(s, a): returns a tuple containing new state and reward, from the input state and action.\n"
                 "\n"
                 "@param instance The Python object instance to call methods on."
        , (arg("self"), "instance")))

        .def("getS",                        &GenerativeModelPython::getS,
                "This function returns the number of states of the world."
        , (arg("self")))

        .def("getA",                        &GenerativeModelPython::getA,
                "This function returns the number of available actions to the agent."
        , (arg("self")))

        .def("getDiscount",                 &GenerativeModelPython::getDiscount,
                "This function returns the currently set discount factor."
        , (arg("self")))

        .def("sampleSR",                    &GenerativeModelPython::sampleSR,
                 "This function samples the MDP for the specified state action pair.\n"
                 "\n"
                 "This function samples the model for simulated experience.\n"
                 "The transition and reward functions are used to produce,\n"
                 "from the state action pair inserted as arguments, a possible\n"
                 "new state with respective reward.  The new state is picked\n"
                 "from all possible states that the MDP allows transitioning\n"
                 "to, each with probability equal to the same probability of\n"
                 "the transition in the model. After a new state is picked,\n"
                 "the reward is the corresponding reward contained in the\n"
                 "reward function.\n"
                 "\n"
                 "@param s The state that needs to be sampled.\n"
                 "@param a The action that needs to be sampled.\n"
                 "\n"
                 "@return A tuple containing a new state and a reward."
        , (arg("self"), "s", "a"))

        .def("isTerminal",                  &GenerativeModelPython::isTerminal,
                "This function returns whether a given state is a terminal."
        , (arg("self"), "s"));
}
