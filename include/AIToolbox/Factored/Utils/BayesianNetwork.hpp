#ifndef AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE
#define AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This struct represents a transition node in a Bayesian network.
     *
     * This struct contains the parents and the transition matrix for a single
     * variable in a BayesianNetwork. Note that the child is not specified, as
     * its id depends on the position of this node within the BayesianNetwork.
     *
     * The number of rows in the matrix correspond to the number of possible
     * combinations of the parents, while the number of columns corresponds to
     * the number of possible values of the child. Each row in the matrix sums
     * up to 1, and every element in it is positive (as the matrix contains
     * conditional probabilities).
     */
    struct BayesianNode {
        PartialKeys tag;
        Matrix2D matrix;
    };

    /**
     * @brief This struct representa a BayesianNetwork.
     *
     * This struct contains a list of BayesianNodes, where each contains the
     * conditional probability table for a single variable. The index of each
     * node represents the index of the variable it is referring to.
     */
    struct BayesianNetwork {
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

        std::vector<BayesianNode> nodes;
    };

    BasisFunction backProject(const Factors & space, const BayesianNetwork & n, const BasisFunction & bf);
    FactoredVector backProject(const Factors & space, const BayesianNetwork & n, const FactoredVector & fv);
}

#endif
