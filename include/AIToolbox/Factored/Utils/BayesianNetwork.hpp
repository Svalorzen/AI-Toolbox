#ifndef AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE
#define AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This class represents the structure of a dynamic decision network.
     *
     * A DDN is a graph that relates how state features and agents are related
     * over a single time steps. In particular, it contains which
     * state-features and agents each next-state-feature depends on.
     *
     * This class constains this information, and allows to compute easily
     * indices to reference outside matrices for data; for example transition
     * probabilities or rewards.
     *
     * This class is supposed to be created once and passed as reference to
     * everybody who needs it, to avoid duplicating information for no reason.
     *
     * This class considers DDNs where the action-parent features are fixed for
     * each next-state-feature, but the state-parent features depend on both
     * the next-state-feature and on what action the parent agents took.
     *
     * For example, if I have a state space
     *
     *     [3, 4, 2]
     *
     * then I have 3 state features. If I have an action space
     *
     *     [2, 5, 4, 2]
     *
     * then I have 4 agents. For each state feature, the DDNGraph has one
     * DDNGraph::Node, so we have 3 of them. Let's assume that the state
     * feature 0 depends on agents 0 and 3; then we will have that, in node 0,
     *
     *     nodes_[0].agents = [0, 3]
     *
     * Now, the space of joint actions for these two agents is 4 (2 * 2). For
     * each one of these, state feature 0 might depend on different sets of
     * state features. So we could have that
     *
     *    nodes_[0].parents = [
     *         [0, 1],     // For joint action value 0,0
     *         [1, 2, 3],  // For joint action value 1,0
     *         [0, 2],     // For joint action value 0,1
     *         [1, 3],     // For joint action value 1,1
     *    ]
     */
    class DynamicDecisionNetworkGraph {
        public:
            /**
             * @brief This class contains the parent information for a single next-state feature.
             */
            struct Node {
                /**
                 * @brief The
                 */
                PartialKeys agents;
                std::vector<PartialKeys> parents;
            };

            /**
             * @brief Basic constructor.
             *
             * Note that in order to be fully initialized, the pushNode(Node&&)
             * method must be called for each state feature.
             *
             * That method is separate to simplify construction and API.
             *
             * \sa pushNode(Node &&);
             *
             * @param S The state space of the DDN.
             * @param A The action space of the DDN.
             */
            DynamicDecisionNetworkGraph(State S, Action A);

            /**
             * @brief This function adds a node to the graph.
             *
             * This method *MUST* be called once per state feature, after
             * construction.
             *
             * This method will sanity check all sets of parents, both agents
             * and state features. Additionally, it will pre-compute the size
             * of each set to speed up the computation of ids.
             *
             * @param node The node to insert.
             */
            void pushNode(Node && node);

            /**
             * @brief This function adds a node to the graph.
             *
             * This method *MUST* be called once per state feature, after
             * construction.
             *
             * This method will sanity check all sets of parents, both agents
             * and state features. Additionally, it will pre-compute the size
             * of each set to speed up the computation of ids.
             *
             * @param node The node to insert.
             */
            void pushNode(const Node & node);

            /**
             * @brief This function computes an id for the input state and action, for the specified feature.
             *
             * \sa getSize(size_t);
             *
             * @param feature The feature to compute the id for.
             * @param s The state to compute the id for.
             * @param a The action to compute the id for.
             *
             * @return A unique id in [0, getSize(feature)).
             */
            size_t getId(size_t feature, const State & s, const Action & a) const;

            /**
             * @brief This function computes an id for the input state and action, for the specified feature.
             *
             * \sa getSize(size_t);
             *
             * @param feature The feature to compute the id for.
             * @param s The state to compute the id for.
             * @param a The action to compute the id for.
             *
             * @return A unique id in [0, getSize(feature)).
             */
            size_t getId(size_t feature, const PartialState & s, const PartialAction & a) const;

            /**
             * @brief This function computes an id from the input action-parent ids, for the specified feature.
             *
             * \sa getIds(size_t, const State &, const Action &);
             * \sa getSize(size_t);
             *
             * @param feature The feature to compute the id for.
             * @param parentId The precomputed parent id.
             * @param actionId The precomputed action id.
             *
             * @return A unique id in [0, getSize(feature)).
             */
            size_t getId(size_t feature, size_t parentId, size_t actionId) const;

            /**
             * @brief This function computes an action and parent ids for the input state, action and feature.
             *
             * This function is provided in case some code still needs to store
             * the "intermediate" ids of the graph.
             *
             * The actionId, which is the first in the pair, is a number
             * between 0 and the factorSpacePartial of the parent agents of the
             * feature.
             *
             * The parentId, which is the second in the pair, is a number
             * between 0 and the factorSpacePartial of the state parent
             * features of the feature, given the input action.
             *
             * \sa getPartialSize(size_t);
             * \sa getPartialSize(size_t, size_t);
             *
             * @param feature The feature to compute the id for.
             * @param s The state to compute the id for.
             * @param a The action to compute the id for.
             *
             * @return A pair of <parent id, action id>.
             */
            std::pair<size_t, size_t> getIds(size_t feature, const State & s, const Action & a) const;

            /**
             * @brief This function computes an action and parent ids for the input state, action and feature.
             *
             * This function is provided in case some code still needs to store
             * the "intermediate" ids of the graph.
             *
             * The actionId, which is the first in the pair, is a number
             * between 0 and the factorSpacePartial of the parent agents of the
             * feature.
             *
             * The parentId, which is the second in the pair, is a number
             * between 0 and the factorSpacePartial of the state parent
             * features of the feature, given the input action.
             *
             * \sa getPartialSize(size_t);
             * \sa getPartialSize(size_t, size_t);
             *
             * @param feature The feature to compute the id for.
             * @param s The state to compute the id for.
             * @param a The action to compute the id for.
             *
             * @return A pair of <parent id, action id>.
             */
            std::pair<size_t, size_t> getIds(size_t feature, const PartialState & s, const PartialAction & a) const;

            /**
             * @brief This function computes an action and parent ids for the input id and feature.
             *
             * This function transform the input "global" id into the action
             * and parent ids.
             *
             * \sa getPartialSize(size_t);
             * \sa getPartialSize(size_t, size_t);
             *
             * @param feature The feature to compute the id for.
             * @param j The global id to decompose.
             *
             * @return A pair of <parent id, action id>.
             */
            std::pair<size_t, size_t> getIds(size_t feature, size_t j);

            /**
             * @brief This function returns the size required to store one element per value of a parent set.
             *
             * Given the input feature, this function returns the number of
             * possible parent sets the feature can have. In other words, it's
             * the sum of the sizes of all state parent features over all
             * possible parent action values.
             *
             * This function is fast as these sizes are pre-computed.
             *
             * @param feature The feature to compute the size for.
             *
             * @return The "global" size of the feature.
             */
            size_t getSize(size_t feature) const;

            /**
             * @brief Thus function returns the number of possible values of the parent agents of the feature.
             *
             * This function is fast as these sizes are pre-computed.
             *
             * @param feature The feature to compute the size for.
             *
             * @return The "action" size of the feature.
             */
            size_t getPartialSize(size_t feature) const;

            /**
             * @brief Thus function returns the number of possible values of the state parent features of the input feature, given the input action id.
             *
             * This function is fast as these sizes are pre-computed.
             *
             * @param feature The feature to compute the size for.
             * @param actionId The id of the action values of the parent agents.
             *
             * @return The "parent" size of the feature, given the parent action.
             */
            size_t getPartialSize(size_t feature, size_t actionId) const;

            /**
             * @brief This function returns the state space of the DDNGraph.
             *
             * @return The state space.
             */
            const State & getS() const;

            /**
             * @brief This function returns the action space of the DDNGraph.
             *
             * @return The action space.
             */
            const Action & getA() const;

            /**
             * @brief This function returns the internal nodes of the DDNGraph.
             *
             * @return The internal nodes.
             */
            const std::vector<Node> & getNodes() const;

        private:
            State S;
            Action A;
            std::vector<Node> nodes_;
            std::vector<std::vector<size_t>> startIds_;
    };

    using DDNGraph = DynamicDecisionNetworkGraph;

    /**
     * @brief This class represents a Dynamic Decision Network with factored actions.
     *
     * This class is able to represent a Dynamic Decision Network where the
     * parents of each factor of the state depend on a particular subset of
     * actions.
     *
     * \sa DynamicDecisionNetworkGraph
     */
    struct DynamicDecisionNetwork {
        using TransitionMatrix = std::vector<Matrix2D>;

        /**
         * @brief This function returns the probability of a transition from one state to another with the given action.
         *
         * @param s The initial factors to start with.
         * @param a The selected action for the transition.
         * @param s1 The factors we should end up with.
         *
         * @return The probability of the transition.
         */
        double getTransitionProbability(const State & s, const Action & a, const State & s1) const;

        /**
         * @brief This function returns the probability of a transition from one state to another.
         *
         * This function allows to compute probabilities for subsets of
         * factors. The parent factors MUST contain all parents of the
         * children!
         *
         * @param s The initial factors to start with.
         * @param a The selected action for the transition.
         * @param s1 The factors we should end up with.
         *
         * @return The probability of the transition.
         */
        double getTransitionProbability(const PartialState & s, const PartialAction & a, const PartialState & s1) const;

        const DDNGraph & graph;
        TransitionMatrix transitions;
    };

    using DDN = DynamicDecisionNetwork;

    BasisMatrix backProject(const DDN & ddn, const BasisFunction & bf);
    FactoredMatrix2D backProject(const DDN & ddn, const FactoredVector & fv);
}

#endif
