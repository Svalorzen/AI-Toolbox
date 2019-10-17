#ifndef AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE
#define AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This class represents a Dynamic Decision Network with factored actions.
     *
     * This class is able to represent a Dynamic Decision Network with factored
     * actions, where the parents of each factor of the state depend on a
     * particular subset of actions.
     */
    struct FactoredDynamicDecisionNetwork {
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
        struct DBNNode {
            PartialKeys tag;
            Matrix2D matrix;
        };

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
            std::vector<DBNNode> nodes;
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

    BasisMatrix backProject(const Factors & space, const Factors & actions, const FactoredDDN & ddn, const BasisFunction & bf);
    FactoredMatrix2D backProject(const Factors & space, const Factors & actions, const FactoredDDN & ddn, const FactoredVector & fv);
}

#endif
