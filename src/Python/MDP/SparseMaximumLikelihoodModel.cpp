#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseMaximumLikelihoodModel.hpp>

#include <boost/python.hpp>

using SparseMaximumLikelihoodModelBinded = AIToolbox::MDP::SparseMaximumLikelihoodModel<AIToolbox::MDP::SparseExperience>;

void exportMDPSparseMaximumLikelihoodModel() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<SparseMaximumLikelihoodModelBinded>{"SparseMaximumLikelihoodModel",

         "@brief This class models Experience as a Markov Decision Process using Maximum Likelihood.\n"
         "\n"
         "Often an MDP is not known in advance. It is known that it can assume\n"
         "a certain set of states, and that a certain set of actions are\n"
         "available to the agent, but not much more. Thus, in these cases, the\n"
         "goal is not only to find out the best policy for the MDP we have,\n"
         "but at the same time learn the actual transition and reward\n"
         "functions of such a model. This task is called 'reinforcement\n"
         "learning'.\n"
         "\n"
         "This class helps with this. A naive approach in reinforcement learning\n"
         "is to keep track, for each action, of its results, and deduce transition\n"
         "probabilities and rewards based on the data collected in such a way.\n"
         "This class does just this, using Maximum Likelihood Estimates to decide\n"
         "what the transition probabilities and rewards are.\n"
         "\n"
         "This class maps an Experience object to the most likely transition\n"
         "reward functions that produced it. The transition function is guaranteed\n"
         "to be a correct probability function, as in the sum of the probabilities\n"
         "of all transitions from a particular state and a particular action is\n"
         "always 1. Each instance is not directly synced with the supplied\n"
         "Experience object. This is to avoid possible overheads, as the user can\n"
         "optimize better depending on their use case. See sync().\n"
         "\n"
         "When little data is available, the deduced transition and reward\n"
         "functions may be significantly subject to noise. A possible way to\n"
         "improve on this is to artificially bias the data as to skew it towards\n"
         "certain distributions.  This could be done if some knowledge of the\n"
         "model (even approximate) is known, in order to speed up the learning\n"
         "process. Another way is to assume that all transitions are possible, add\n"
         "data to support that claim, and simply wait until the averages converge\n"
         "to the true values.  Another thing that can be done is to associate with\n"
         "each fake datapoint an high reward: this will skew the agent into trying\n"
         "out new actions, thinking it will obtained the high rewards. This is\n"
         "able to obtain automatically a good degree of exploration in the early\n"
         "stages of an episode. Such a technique is called 'optimistic\n"
         "initialization'.\n"
           "\n"
         "Whether any of these techniques work or not can definitely depend on\n"
         "the model you are trying to approximate. Trying out things is good!\n"
           "\n"
         "The difference between this class and the MDP.MaximumLikelihoodModel class is that\n"
         "this class stores transitions and rewards in sparse matrices. This\n"
         "results in a possibly slower access to individual probabilities and\n"
         "rewards, but immeasurably speeds up computation with some classes of\n"
         "planning algorithms in case the number of useful transitions is very\n"
         "small with respect to the total theoretic state action space of\n"
         "SxAxS. It also of course incredibly reduces memory consumption in\n"
         "such cases, which may also improve speed by effect of improved\n"
         "caching.\n", no_init}

        .def(init<const SparseExperience &, optional<double, bool>>(
                 "Constructor using previous Experience.\n"
                   "\n"
                 "This constructor selects the Experience that will\n"
                 "be used to learn an MDP Model from the data, and initializes\n"
                 "internal Model data.\n"
                   "\n"
                 "The user can choose whether he wants to directly sync\n"
                 "the SparseMaximumLikelihoodModel to the underlying Experience,\n"
                 "or delay it for later.\n"
                   "\n"
                 "In the latter case the default transition function\n"
                 "defines a transition of probability 1 for each\n"
                 "state to itself, no matter the action.\n"
                   "\n"
                 "In general it would be better to add some amount of bias\n"
                 "to the Experience so that when a new state-action pair is\n"
                 "tried, the SparseMaximumLikelihoodModel doesn't automatically\n"
                 "compute 100% probability of transitioning to the resulting state,\n"
                 "but smooths into it. This may depend on your problem though.\n"
                   "\n"
                 "The default reward function is 0.\n"
                   "\n"
                 "@param exp The base Experience of the model.\n"
                 "@param discount The discount used in solving methods.\n"
                 "@param sync Whether to sync with the Experience immediately or delay it."
        , (arg("self"), "exp", "discount", "sync")))

        .def("setDiscount",                 &SparseMaximumLikelihoodModelBinded::setDiscount,
                 "This function sets a new discount factor for the Model.\n"
        , (arg("self"), "discount"))

        .def("sync",                        static_cast<void(SparseMaximumLikelihoodModelBinded::*)()>(&SparseMaximumLikelihoodModelBinded::sync),
                 "This function syncs the whole SparseMaximumLikelihoodModel to the underlying Experience.\n"
                 "\n"
                 "Since use cases in AI are very varied, one may not want to update\n"
                 "its SparseMaximumLikelihoodModel for each single transition experienced\n"
                 "by the agent. To avoid this we leave to the user the task of syncing between the\n"
                 "underlying Experience and the SparseMaximumLikelihoodModel, as he/she sees fit.\n"
                 "\n"
                 "After this function is run the transition and reward functions\n"
                 "will accurately reflect the state of the underlying Experience."
        , (arg("self")))

        .def("sync",                        static_cast<void(SparseMaximumLikelihoodModelBinded::*)(size_t, size_t)>(&SparseMaximumLikelihoodModelBinded::sync),
                 "This function syncs a state action pair in the SparseMaximumLikelihoodModel to the underlying Experience.\n"
                 "\n"
                 "Since use cases in AI are very varied, one may not want to update\n"
                 "its SparseMaximumLikelihoodModel for each single transition experienced\n"
                 "by the agent. To avoid this we leave to the user the task of syncing between the\n"
                 "underlying Experience and the SparseMaximumLikelihoodModel, as he/she sees fit.\n"
                 "\n"
                 "This function updates a single state action pair with the underlying\n"
                 "Experience. This function is offered to avoid having to recompute the\n"
                 "whole SparseMaximumLikelihoodModel if the user knows that only few transitions\n"
                 "have been experienced by the agent.\n"
                 "\n"
                 "After this function is run the transition and reward functions\n"
                 "will accurately reflect the state of the underlying Experience\n"
                 "for the specified state action pair.\n"
                 "\n"
                 "@param s The state that needs to be synced.\n"
                 "@param a The action that needs to be synced."
        , (arg("self"), "s", "a"))

        .def("sync",                        static_cast<void(SparseMaximumLikelihoodModelBinded::*)(size_t, size_t, size_t)>(&SparseMaximumLikelihoodModelBinded::sync),
                 "This function syncs a state action pair in the SparseMaximumLikelihoodModel to the underlying Experience in the fastest possible way.\n"
                 "\n"
                 "This function updates a state action pair given that the last increased transition\n"
                 "in the underlying Experience is the triplet s, a, s1. In addition, this function only\n"
                 "works if it needs to add information from this single new point of information (if\n"
                 "more has changed from the last sync, use sync(s,a) ). The performance boost that\n"
                 "this function obtains increases with the increase of the number of states in the model.\n"
                 "\n"
                 "@param s The state that needs to be synced.\n"
                 "@param a The action that needs to be synced.\n"
                 "@param s1 The final state of the transition that got updated in the Experience."
        , (arg("self"), "s", "a", "s1"))

        .def("sampleSR",                    &SparseMaximumLikelihoodModelBinded::sampleSR,
                 "This function samples the MDP for the specified state action pair.\n"
                 "\n"
                 "This function samples the model for simulate experience. The transition\n"
                 "and reward functions are used to produce, from the state action pair\n"
                 "inserted as arguments, a possible new state with respective reward.\n"
                 "The new state is picked from all possible states that the MDP allows\n"
                 "transitioning to, each with probability equal to the same probability\n"
                 "of the transition in the model. After a new state is picked, the reward\n"
                 "is the corresponding reward contained in the reward function.\n"
                 "\n"
                 "@param s The state that needs to be sampled.\n"
                 "@param a The action that needs to be sampled.\n"
                 "\n"
                 "@return A tuple containing a new state and a reward."
        , (arg("self"), "s", "a"))

        .def("getS",                        &SparseMaximumLikelihoodModelBinded::getS,
                 "This function returns the number of states of the world."
        , (arg("self")))

        .def("getA",                        &SparseMaximumLikelihoodModelBinded::getA,
                 "This function returns the number of available actions to the agent."
        , (arg("self")))

        .def("getDiscount",                 &SparseMaximumLikelihoodModelBinded::getDiscount,
                 "This function returns the currently set discount factor."
        , (arg("self")))

        .def("getExperience",               &SparseMaximumLikelihoodModelBinded::getExperience, return_value_policy<reference_existing_object>(),
                 "This function enables inspection of the underlying Experience of the SparseMaximumLikelihoodModel."
        , (arg("self")))

        .def("getTransitionProbability",    &SparseMaximumLikelihoodModelBinded::getTransitionProbability,
                 "This function returns the stored transition probability for the specified transition."
        , (arg("self"), "s", "a", "s1"))

        .def("getExpectedReward",           &SparseMaximumLikelihoodModelBinded::getExpectedReward,
                 "This function returns the stored expected reward for the specified transition."
        , (arg("self"), "s", "a", "s1"))

        .def("isTerminal",                  &SparseMaximumLikelihoodModelBinded::isTerminal,
                 "This function returns whether a given state is a terminal."
        , (arg("self"), "s"));
}
