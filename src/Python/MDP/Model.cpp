#include <AIToolbox/MDP/Model.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportMDPModel() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<Model>{"Model",

         "This class represents a Markov Decision Process.\n"
         "\n"
         "A Markov Decision Process (MDP) is a way to model decision making.\n"
         "The idea is that there is an agent situated in a stochastic\n"
         "environment which changes in discrete 'timesteps'. The agent can\n"
         "influence the way the environment changes via 'actions'. For each\n"
         "action the agent can perform, the environment will transition from a\n"
         "state 's' to a state 's1' following a certain transition function.\n"
         "The transition function specifies, for each triple SxAxS' the\n"
         "probability that such a transition will happen.\n"
         "\n"
         "In addition, associated with transitions, the agent is able to\n"
         "obtain rewards. Thus, if it does good, the agent will obtain a\n"
         "higher reward than if it performed badly. The reward obtained by the\n"
         "agent is in addition associated with a 'discount' factor: at every\n"
         "step, the possible reward that the agent can collect is multiplied\n"
         "by this factor, which is a number between 0 and 1. The discount\n"
         "factor is used to model the fact that often it is preferable to\n"
         "obtain something sooner, rather than later.\n"
         "\n"
         "Since all of this is governed by probabilities, it is possible to\n"
         "solve an MDP model in order to obtain an 'optimal policy', which is\n"
         "a way to select an action from a state which will maximize the\n"
         "expected reward that the agent is going to collect during its life.\n"
         "The expected reward is computed as the sum of every reward the agent\n"
         "collects at every timestep, keeping in mind that at every timestep\n"
         "the reward is further and further discounted.\n"
         "\n"
         "Solving an MDP in such a way is called 'planning'. Planning\n"
         "solutions often include an 'horizon', which is the number of\n"
         "timesteps that are included in an episode. They can be finite or\n"
         "infinite. The optimal policy changes with respect to the horizon,\n"
         "since a higher horizon may offer access to reward-gaining\n"
         "opportunities farther in the future.\n"
         "\n"
         "An MDP policy (be it the optimal one or another), is associated with\n"
         "two functions: a ValueFunction and a QFunction. The ValueFunction\n"
         "represents the expected return for the agent from any initial state,\n"
         "given that actions are going to be selected according to the policy.\n"
         "The QFunction is similar: it gives the expected return for a\n"
         "specific state-action pair, given that after the specified action\n"
         "one will act according to the policy.\n"
         "\n"
         "Given that we are usually interested about the optimal policy, there\n"
         "are a couple of properties that are associated with the optimal\n"
         "policies functions.  First, the optimal policy can be derived from\n"
         "the optimal QFunction. The optimal policy simply selects, in a given\n"
         "state 's', the action that maximizes the value of the QFunction.  In\n"
         "the same way, the optimal ValueFunction can be computed from the\n"
         "optimal QFunction by selecting the max with respect to the action.\n"
         "\n"
         "Since so much information can be extracted from the QFunction, lots\n"
         "of methods (mostly in Reinforcement Learning) try to learn it.", no_init}

        .def(init<size_t, size_t, optional<double>>(
                 "Basic constructor.\n"
                 "\n"
                 "This constructor initializes the Model so that all\n"
                 "transitions happen with probability 0 but for transitions\n"
                 "that bring back to the same state, no matter the action.\n"
                 "\n"
                 "All rewards are set to 0. The discount parameter is set to\n"
                 "1.\n"
                 "\n"
                 "@param s The number of states of the world.\n"
                 "@param a The number of actions available to the agent.\n"
                 "@param discount The discount factor for the MDP."
        , (arg("self"), "s", "a", "discount")))

        .def(init<const Model &>(
                 "This allows to copy from any other model. A nice use for this is to\n"
                 "convert any model which computes probabilities on the fly into an\n"
                 "MDP::Model where probabilities are all stored for fast access. Of\n"
                 "course such a solution can be done only when the number of states\n"
                 "and actions is not too big."
        , (arg("self"), "model")))

        .def(init<const SparseModel &>(
                 "This allows to copy from any other model. A nice use for this is to\n"
                 "convert any model which computes probabilities on the fly into an\n"
                 "MDP::Model where probabilities are all stored for fast access. Of\n"
                 "course such a solution can be done only when the number of states\n"
                 "and actions is not too big."
        , (arg("self"), "sparseModel")))

        .def(init<const RLModel<Experience> &>(
                 "This allows to copy from any other model. A nice use for this is to\n"
                 "convert any model which computes probabilities on the fly into an\n"
                 "MDP::Model where probabilities are all stored for fast access. Of\n"
                 "course such a solution can be done only when the number of states\n"
                 "and actions is not too big."
        , (arg("self"), "rlModel")))

        .def(init<const SparseRLModel<SparseExperience> &>(
                 "This allows to copy from any other model. A nice use for this is to\n"
                 "convert any model which computes probabilities on the fly into an\n"
                 "MDP::Model where probabilities are all stored for fast access. Of\n"
                 "course such a solution can be done only when the number of states\n"
                 "and actions is not too big."
        , (arg("self"), "sparseRLModel")))

        .def("setDiscount",                 &Model::setDiscount,
                "This function sets a new discount factor for the Model."
        , (arg("self"), "discount"))

        .def("setTransitionFunction",       &Model::setTransitionFunction<std::vector<std::vector<std::vector<double>>>>,
                "This function replaces the Model transition function with the one provided.\n"
                "\n"
                "Currently the Python wrappings support reading through native 3d Python\n"
                "arrays (so [][][]). As long as the dimensions are correct and they contain\n"
                "correct probabilities everything should be fine. The code should reject\n"
                "them otherwise."
        , (arg("self"), "transitionFunction3D"))

        .def("setRewardFunction",           &Model::setRewardFunction<std::vector<std::vector<std::vector<double>>>>,
                "This function replaces the Model reward function with the one provided.\n"
                "\n"
                "Currently the Python wrappings support reading through native 3d Python\n"
                "arrays (so [][][]). As long as the dimensions are correct and they contain\n"
                "correct probabilities everything should be fine. The code should reject\n"
                "them otherwise."
        , (arg("self"), "rewardFunction3D"))

        .def("getS",                        &Model::getS,
                "This function returns the number of states of the world."
        , (arg("self")))

        .def("getA",                        &Model::getA,
                "This function returns the number of available actions to the agent."
        , (arg("self")))

        .def("getDiscount",                 &Model::getDiscount,
                "This function returns the currently set discount factor."
        , (arg("self")))

        .def("sampleSR",                    &Model::sampleSR,
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

        .def("getTransitionProbability",    &Model::getTransitionProbability,
                "This function returns the stored transition probability for the specified transition."
        , (arg("self"), "s", "a", "s1"))

        .def("getExpectedReward",           &Model::getExpectedReward,
                "This function returns the stored expected reward for the specified transition."
        , (arg("self"), "s", "a", "s1"))

        .def("isTerminal",                  &Model::isTerminal,
                "This function returns whether a given state is a terminal."
        , (arg("self"), "s"));
}
