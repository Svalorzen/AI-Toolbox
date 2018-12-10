#ifndef AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE
#define AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
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
     *
     * This class can either own the nodes, or just store references to them.
     * This is because we can generate a BayesianNetwork on the fly from a
     * default transition model and a set of BayesianDiffNodes. The resulting
     * BayesianNetwork takes into account of the differences between the diff
     * and the default, and substitutes the changed nodes in the final result.
     *
     * @tparam UseReference Whether we want to store references to BayesianNodes rather than own the nodes.
     */
    template <bool UseReference>
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

        /**
         * @brief This function returns a reference to the ith BayesianNode in the network.
         *
         * This can be used to bypass the reference wrappers easily.
         */
        const BayesianNode & operator[](size_t i) const;

        using Node = std::conditional_t<UseReference, std::reference_wrapper<const BayesianNode>, BayesianNode>;

        std::vector<Node> nodes;
    };

    BayesianNetwork() -> BayesianNetwork<false>;
    BayesianNetwork(std::vector<BayesianNode>) -> BayesianNetwork<false>;

    /**
     * @brief This struct allows to change a default transition model in a compact manner.
     *
     * As we use BayesianNetworks in order to contain factored transition
     * functions, each action usually denotes a separate network. However, the
     * networks are usually similar, as each action only affects a subset of
     * the states.
     *
     * This struct allows to define compactly such differences, by specifying
     * only the nodes that are different from the default transition model.
     */
    struct BayesianDiffNode {
        size_t id;
        BayesianNode node;
    };

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
    class ParametricBayesianNetwork {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param diffs The differences for each parameter from the default transition.
             * @param defaultTransition The default transition model.
             */
            ParametricBayesianNetwork(
                std::vector<std::vector<BayesianDiffNode>> diffs,
                BayesianNetwork<false> defaultTransition
            );

            /**
             * @brief This function returns the default transition model.
             *
             * @return The default transition model.
             */
            const BayesianNetwork<false> & getDefaultTransition() const;

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
            BayesianNetwork<true> makeDiffTransition(const size_t a) const;

        private:
            std::vector<std::vector<BayesianDiffNode>> diffs_;
            BayesianNetwork<false> defaultTransition_;
    };

    template <bool Ref>
    double BayesianNetwork<Ref>::getTransitionProbability(const Factors & space, const Factors & s, const Factors & s1) const {
        double retval = 1.0;

        // For each partial transition matrix, we compute the entry which
        // applies to this transition, and we multiply all entries together.
        for (size_t i = 0; i < space.size(); ++i) {
            // Compute parent ID based on the parents of state factor 'i'
            auto parentId = toIndexPartial((*this)[i].tag, space, s);
            retval *= (*this)[i].matrix(parentId, s1[i]);
        }

        return retval;
    }

    template <bool Ref>
    double BayesianNetwork<Ref>::getTransitionProbability(const Factors & space, const PartialFactors & s, const PartialFactors & s1) const {
        double retval = 1.0;
        // The matrix is made up of one component per child, and we
        // need to multiply all of them together. At each iteration we
        // look at a different "child".
        for (size_t j = 0; j < s1.first.size(); ++j) {
            // Find the matrix relative to this child
            const auto & node = (*this)[s1.first[j]];
            // Compute the "dense" id for the needed parents
            // from the current domain.
            auto id = toIndexPartial(node.tag, space, s);
            // Multiply the current value by the lhs value.
            retval *= node.matrix(id, s1.second[j]);
        }
        return retval;
    }

    template <bool Ref>
    const BayesianNode & BayesianNetwork<Ref>::operator[](size_t i) const {
        if constexpr (Ref) {
            return nodes[i].get();
        } else {
            return nodes[i];
        }
    }

    template <bool Ref>
    BasisFunction backProject(const Factors & space, const BayesianNetwork<Ref> & b, const BasisFunction & bf) {
        // Here we have the two function inputs, in this form:
        //
        //     lhs: [parents, child] -> value
        //     rhs: [children] -> value
        BasisFunction retval;

        // The domain here depends on the parents of all elements of
        // the domain of the input basis.
        for (auto d : bf.tag)
            retval.tag = merge(retval.tag, b[d].tag);

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
                currentVal += bf.values[i] * b.getTransitionProbability(space, *domain, *rhsdomain);

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

    template <bool Ref>
    FactoredVector backProject(const Factors & space, const BayesianNetwork<Ref> & b, const FactoredVector & fv) {
        FactoredVector retval;
        retval.bases.reserve(fv.bases.size());

        for (const auto & basis : fv.bases) {
            plusEqual(space, retval,
                    backProject(space, b, basis));
        }

        return retval;
    }
}

#endif
