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
     * conditional probability table for a single variable. The index of each
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
     * @brief This class represents compactly a set of similar BayesianNetworks.
     *
     * This class allows to represent in a compact manner a set of
     * BayesianNetworks that all closely resemble a default transition model.
     *
     * The default transition model is stored together with a set of
     * differences, which are applied on the fly to produce the correct
     * BayesianNetwork for every possible parameter.
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
             * @param diffs The differences for each parameter from the default transition.
             * @param defaultTransition The default transition model.
             */
            CompactDynamicDecisionNetwork(
                std::vector<std::vector<Node>> diffs,
                DBN defaultTransition
            );

            /**
             * @brief This function construct a BayesianNetwork for the specified parameter.
             *
             * The output is a network that contains references to nodes owned
             * by this class. Thus it is (relatively) cheap to create and to
             * copy, but its lifetime depends on the instance that created it.
             *
             * @param a The parameter selecting the BayesianNetwork to create.
             *
             * @return The BayesianNetwork for the specified parameter.
             */
            DBNRef makeDiffTransition(const size_t a) const;

            /**
             * @brief This function returns the default transition model.
             *
             * @return The default transition model.
             */
            const DBN & getDefaultTransition() const;

            /**
             * @brief This function returns the diff nodes for this ParametricBayesianNetwork.
             */
            const std::vector<std::vector<Node>> & getDiffNodes() const;

        private:
            std::vector<std::vector<Node>> diffs_;
            DBN defaultTransition_;
    };

    using CompactDDN = CompactDynamicDecisionNetwork;

    template <typename Net>
    BasisFunction backProject(const Factors & space, const Net & dbn, const BasisFunction & bf) {
        // Here we have the two function inputs, in this form:
        //
        //     lhs: [parents, child] -> value
        //     rhs: [children] -> value
        BasisFunction retval;

        // The domain here depends on the parents of all elements of
        // the domain of the input basis.
        for (auto d : bf.tag)
            retval.tag = merge(retval.tag, dbn[d].tag);

        retval.values.resize(factorSpacePartial(retval.tag, space));
        // Don't need to zero fill

        // Iterate over the domain, since the output basis is going to
        // be dense pretty much.
        size_t id = 0;
        PartialFactorsEnumerator domain(space, retval.tag);

        PartialFactorsEnumerator rhsdomain(space, bf.tag);
        // PartialFactorsEnumerator rhsdomain(space);

        while (domain.isValid()) {
            // For each domain assignment, we need to go over every
            // possible children assignment. As we are computing
            // products, it is sufficient to go over the elements
            // stored in the RHS (as all other children combinations
            // are zero by definition).
            //
            // For each such assignment, we compute the product of the
            // rhs there with the value of the lhs at the current
            // domain & children.
            double currentVal = 0.0;
            size_t i = 0;
            while (rhsdomain.isValid()) {
                currentVal += bf.values[i] * dbn.getTransitionProbability(space, *domain, *rhsdomain);

                ++i;
                rhsdomain.advance();
            }
            retval.values[id] = currentVal;

            ++id;
            domain.advance();
            rhsdomain.reset();
        }
        return retval;
    }

    template <typename Net>
    FactoredVector backProject(const Factors & space, const Net & dbn, const FactoredVector & fv) {
        FactoredVector retval;
        retval.bases.reserve(fv.bases.size());

        for (const auto & basis : fv.bases) {
            plusEqual(space, retval,
                    backProject(space, dbn, basis));
        }

        return retval;
    }
}

#endif
