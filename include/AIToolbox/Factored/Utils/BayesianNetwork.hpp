#ifndef AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE
#define AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This struct represents a Dynamic Bayesian Network.
     *
     * This struct contains a list of DynamicBayesianNodes, where each contains the
     * conditional probability matrix for a single variable. The index of each
     * node represents the index of the variable it is referring to.
     */
    struct DynamicBayesianNetwork {
        /**
         * @brief This struct represents a transition node in a Dynamic Bayesian network.
         *
         * This struct contains the parents and the transition matrix for a single
         * variable in a DynamicBayesianNetwork. Note that the child is not
         * specified, as its id depends on the position of this node within the
         * DynamicBayesianNetwork.
         *
         * The number of rows in the matrix correspond to the number of possible
         * combinations of the parents, while the number of columns corresponds to
         * the number of possible values of the child. Each row in the matrix sums
         * up to 1, and every element in it is positive (as the matrix contains
         * conditional probabilities).
         */
        struct Node {
            PartialKeys tag;
            Matrix2D matrix;
        };

        /**
         * @brief This function returns the probability of a transition from one state to another.
         *
         * @param space The factor space to use.
         * @param s The initial factors to start with.
         * @param s1 The factors we should end up with.
         *
         * @return The probability of the transition.
         */
        double getTransitionProbability(const Factors & space, const Factors & s, const Factors & s1) const;

        /**
         * @brief This function returns the probability of a transition from one state to another.
         *
         * This function allows to compute probabilities for subsets of
         * factors. The initial factors MUST contain all parents of the
         * children!
         *
         * @param space The factor space to use.
         * @param s The initial factors to start with.
         * @param s1 The factors we should end up with.
         *
         * @return The probability of the transition.
         */
        double getTransitionProbability(const Factors & space, const PartialFactors & s, const PartialFactors & s1) const;

        /**
         * @brief This function returns a reference to the ith DynamicBayesianNode in the network.
         *
         * This is useful to write template code that uses both this and DynamicBayesianNetworkRef.
         */
        const Node & operator[](size_t i) const;

        std::vector<Node> nodes;
    };

    using DBN = DynamicBayesianNetwork;

    /**
     * @brief This struct is a non-owning Dynamic Bayesian Network.
     *
     * This class is useful to create DBNs on the fly from pre-existing
     * DBN::Nodes, without the need to copy them. The interface is exactly the
     * same as a DynamicBayesianNetwork, with the only difference that it
     * stores std::reference_wrappers to the DBN::Nodes.
     */
    struct DynamicBayesianNetworkRef {
        /**
         * @brief This function returns the probability of a transition from one state to another.
         *
         * @param space The factor space to use.
         * @param s The initial factors to start with.
         * @param s1 The factors we should end up with.
         *
         * @return The probability of the transition.
         */
        double getTransitionProbability(const Factors & space, const Factors & s, const Factors & s1) const;

        /**
         * @brief This function returns the probability of a transition from one state to another.
         *
         * This function allows to compute probabilities for subsets of
         * factors. The initial factors MUST contain all parents of the
         * children!
         *
         * @param space The factor space to use.
         * @param s The initial factors to start with.
         * @param s1 The factors we should end up with.
         *
         * @return The probability of the transition.
         */
        double getTransitionProbability(const Factors & space, const PartialFactors & s, const PartialFactors & s1) const;

        /**
         * @brief This function returns a reference to the ith DynamicBayesianNode in the network.
         *
         * This is useful to write template code that uses both this and DynamicBayesianNetwork.
         */
        const DBN::Node & operator[](size_t i) const;

        std::vector<std::reference_wrapper<const DBN::Node>> nodes;
    };

    using DBNRef = DynamicBayesianNetworkRef;

    /**
     * @brief This class represents a Dynamic Decision Network compactly.
     *
     * This class allows to represent in a compact manner a set of
     * DynamicBayesianNetworks that all closely resemble a default transition
     * model.
     *
     * The default transition model is stored together with a set of
     * differences - one per action. When the DynamicBayesianNetwork for a
     * particular action is requested, the correct diffs are applied on the fly
     * to produce the correct DynamicBayesianNetwork.
     *
     * We actually produce a DynamicBayesianNetworkRef which contains
     * references to the nodes, so that the construction does not require too
     * much time nor space.
     */
    class CompactDynamicDecisionNetwork {
        public:
            /**
             * @brief This struct allows to change a default transition model in a compact manner.
             *
             * As we use DynamicBayesianNetworks in order to contain factored transition
             * functions, each action usually denotes a separate network. However, the
             * networks are usually similar, as each action only affects a subset of
             * the states.
             *
             * This struct allows to define compactly such differences, by specifying
             * only the nodes that are different from the default transition model.
             */
            struct Node {
                size_t id;
                DBN::Node node;
            };

            /**
             * @brief Basic constructor.
             *
             * @param diffs The differences for each action from the default transition.
             * @param defaultTransition The default transition model.
             */
            CompactDynamicDecisionNetwork(
                std::vector<std::vector<Node>> diffs,
                DBN defaultTransition
            );

            /**
             * @brief This function constructs a DynamicBayesianNetworkRef for the specified action.
             *
             * The output is a network that contains references to nodes owned
             * by this class. Thus it is (relatively) cheap to create and to
             * copy, but its lifetime depends on the instance that created it.
             *
             * @param a The desired action to use.
             *
             * @return The DynamicBayesianNetworkRef for the specified action.
             */
            DBNRef makeDiffTransition(const size_t a) const;

            /**
             * @brief This function returns the default transition model.
             *
             * @return The default transition model.
             */
            const DBN & getDefaultTransition() const;

            /**
             * @brief This function returns the diff nodes for this CompactDynamicDecisionNetwork.
             */
            const std::vector<std::vector<Node>> & getDiffNodes() const;

        private:
            std::vector<std::vector<Node>> diffs_;
            DBN defaultTransition_;
    };

    using CompactDDN = CompactDynamicDecisionNetwork;

    /**
     * @brief This class represents a Dynamic Decision Network with factored actions.
     *
     * This class is able to represent a Dynamic Decision Network with factored
     * actions, where the parents of each factor of the state depend on a
     * particular subset of actions.
     */
    struct FactoredDynamicDecisionNetwork {
        /**
         * @brief This struct contains the transition matrices for a particular factor.
         *
         * As the parents of each factor depend on a subset of actions,
         * this struct contains the indeces of the factored actions that
         * are needed in order to determine the parents, and a list
         * containing a DBN::Node for every possible action combination.
         */
        struct Node {
            PartialKeys actionTag;
            std::vector<DBN::Node> nodes;
        };

        /**
         * @brief This function returns the probability of a transition from one state to another with the given action.
         *
         * @param space The factor space to use.
         * @param actions The action space to use.
         * @param s The initial factors to start with.
         * @param a The selected action for the transition.
         * @param s1 The factors we should end up with.
         *
         * @return The probability of the transition.
         */
        double getTransitionProbability(const Factors & space, const Factors & actions, const Factors & s, const Factors & a, const Factors & s1) const;

        /**
         * @brief This function returns the probability of a transition from one state to another.
         *
         * This function allows to compute probabilities for subsets of
         * factors. The initial factors MUST contain all parents of the
         * children!
         *
         * @param space The factor space to use.
         * @param actions The action space to use.
         * @param s The initial factors to start with.
         * @param a The selected action for the transition.
         * @param s1 The factors we should end up with.
         *
         * @return The probability of the transition.
         */
        double getTransitionProbability(const Factors & space, const Factors & actions, const PartialFactors & s, const PartialFactors & a, const PartialFactors & s1) const;

        /**
         * @brief This function returns a reference to the ith DynamicBayesianNode in the network.
         *
         * This is useful to write template code that uses both this and DynamicBayesianNetwork.
         */
        const Node & operator[](size_t i) const;

        std::vector<Node> nodes;
    };

    using FactoredDDN = FactoredDynamicDecisionNetwork;

    BasisFunction backProject(const Factors & space, const DBN & dbn, const BasisFunction & bf);
    BasisFunction backProject(const Factors & space, const DBNRef & dbn, const BasisFunction & bf);
    FactoredVector backProject(const Factors & space, const DBN & dbn, const FactoredVector & fv);
    FactoredVector backProject(const Factors & space, const DBNRef & dbn, const FactoredVector & fv);
    BasisMatrix backProject(const Factors & space, const Factors & actions, const FactoredDDN & ddn, const BasisFunction & bf);
    FactoredMatrix2D backProject(const Factors & space, const Factors & actions, const FactoredDDN & ddn, const FactoredVector & fv);
}

#endif
