#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <boost/python.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPModel() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<POMDPModelBinded, bases<AIToolbox::MDP::Model>>{"Model",

        "This class represents a Partially Observable Markov Decision Process.\n"
        "\n"
        "This class inherits from any valid MDP model type, so that it can\n"
        "use its base methods, and it builds from those. Templated inheritance\n"
        "was chosen to improve performance and keep code small, instead of\n"
        "doing composition.\n"
        "\n"
        "A POMDP is an MDP where the agent, at each timestep, does not know\n"
        "in which state it is. Instead, after each action is performed, it\n"
        "obtains an 'observation', which offers some information as to which\n"
        "new state the agent has transitioned to. This observation is\n"
        "determined by an 'observation function', that maps S'xAxO to a\n"
        "probability: the probability of obtaining observation O after taking\n"
        "action A and *landing* in state S'.\n"
        "\n"
        "Since now its knowledge is imperfect, in order to represent the\n"
        "knowledge of the state it is currently in, the agent is thus forced\n"
        "to use Beliefs: probability distributions over states.\n"
        "\n"
        "The way a Belief works is that, after each action and observation,\n"
        "the agent can reason as follows: given my previous Belief\n"
        "(distribution over states) that I think I was in, what is now the\n"
        "probability that I transitioned to any particular state? This new\n"
        "Belief can be computed from the Model, given that the agent knows\n"
        "the distributions of the transition and observation functions.\n"
        "\n"
        "Turns out that a POMDP can be viewed as an MDP with an infinite\n"
        "number of states, where each state is essentially a Belief. Since a\n"
        "Belief is a vector of real numbers, there are infinite of them, thus\n"
        "the infinite number of states. While POMDPs can be much more\n"
        "powerful than MDPs for modeling real world problems, where\n"
        "information is usually not perfect, it turns out that this\n"
        "infinite-state property makes them so much harder to solve\n"
        "perfectly, and their solutions much more complex.\n"
        "\n"
        "A POMDP solution is composed by several policies, which apply in\n"
        "different ranges of the Belief space, and suggest different actions\n"
        "depending on the observations received by the agent at each\n"
        "timestep. The values of those policies can be, in the same way,\n"
        "represented as a number of value vectors (called alpha vectors in\n"
        "the literature) that apply in those same ranges of the Belief space.\n"
        "Each alpha vector is somewhat similar to an MDP ValueFunction.", no_init}

        .def(init<size_t, size_t, size_t, optional<double>>(
                "Basic constructor.\n"
                "\n"
                "This constructor initializes the observation function\n"
                "so that all actions will return observation 0.\n"
                "\n"
                "This constructor initializes the Model so that all\n"
                "transitions happen with probability 0 but for transitions\n"
                "that bring back to the same state, no matter the action.\n"
                "\n"
                "All rewards are set to 0. The discount parameter is set to\n"
                "1.\n"
                "\n"
                "@param o The number of possible observations the agent could make.\n"
                "@param s The number of states of the world.\n"
                "@param a The number of actions available to the agent.\n"
                "@param discount The discount factor for the MDP.\n"
        , (arg("self"), "o", "s", "a", "discount")))

        .def(init<const POMDPModelBinded &>(
                 "This allows to copy from any other model. A nice use for this is to\n"
                 "convert any model which computes probabilities on the fly into an\n"
                 "MDP::Model where probabilities are all stored for fast access. Of\n"
                 "course such a solution can be done only when the number of states\n"
                 "and actions is not too big."
        , (arg("self"), "model")))

        .def(init<const POMDPSparseModelBinded &>(
                 "This allows to copy from any other model. A nice use for this is to\n"
                 "convert any model which computes probabilities on the fly into an\n"
                 "MDP::Model where probabilities are all stored for fast access. Of\n"
                 "course such a solution can be done only when the number of states\n"
                 "and actions is not too big."
        , (arg("self"), "sparseModel")))

        .def("setObservationFunction",      &POMDPModelBinded::setObservationFunction<std::vector<std::vector<std::vector<double>>>>,
                "This function replaces the Model observation function with the one provided.\n"
                "\n"
                "Currently the Python wrappings support reading through native 3d Python\n"
                "arrays (so [][][]). As long as the dimensions are correct and they contain\n"
                "correct probabilities everything should be fine. The code should reject\n"
                "them otherwise."
        , (arg("self"), "observationFunction3D"))

        .def("sampleSOR",                   &POMDPModelBinded::sampleSOR,
                "This function samples the POMDP for the specified state action pair.\n"
                "\n"
                "This function samples the model for simulated experience. The\n"
                "transition, observation and reward functions are used to\n"
                "produce, from the state action pair inserted as arguments, a\n"
                "possible new state with respective observation and reward.\n"
                "The new state is picked from all possible states that the\n"
                "MDP allows transitioning to, each with probability equal to\n"
                "the same probability of the transition in the model. After a\n"
                "new state is picked, an observation is sampled from the\n"
                "observation function distribution, and finally the reward is\n"
                "the corresponding reward contained in the reward function.\n"
                "\n"
                "@param s The state that needs to be sampled.\n"
                "@param a The action that needs to be sampled.\n"
                "\n"
                "@return A tuple containing a new state, observation and reward."
        , (arg("self"), "s", "a"))

        .def("sampleOR",                    &POMDPModelBinded::sampleOR,
                "This function samples the POMDP for the specified state action pair.\n"
                "\n"
                "This function samples the model for simulated experience.\n"
                "The transition, observation and reward functions are used to\n"
                "produce, from the state, action and new state inserted as\n"
                "arguments, a possible new observation and reward. The\n"
                "observation and rewards are picked so that they are\n"
                "consistent with the specified new state.\n"
                "\n"
                "@param s The state that needs to be sampled.\n"
                "@param a The action that needs to be sampled.\n"
                "@param s1 The resulting state of the s,a transition.\n"
                "\n"
                "@return A tuple containing a new observation and reward."
        , (arg("self"), "s", "a", "s1"))

        .def("getO",                        &POMDPModelBinded::getO,
                 "This function returns the number of observations possible."
        , (arg("self")))

        .def("getObservationProbability",   &POMDPModelBinded::getObservationProbability,
                 "This function returns the stored observation probability for the specified state-action pair."
        , (arg("self"), "s", "a", "s1"));
}
