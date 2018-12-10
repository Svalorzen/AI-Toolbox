#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    double BayesianNetwork::getTransitionProbability(const Factors & space, const Factors & s, const Factors & s1) const {
        double retval = 1.0;

        // For each partial transition matrix, we compute the entry which
        // applies to this transition, and we multiply all entries together.
        for (size_t i = 0; i < space.size(); ++i) {
            // Compute parent ID based on the parents of state factor 'i'
            auto parentId = toIndexPartial(nodes[i].tag, space, s);
            retval *= nodes[i].matrix(parentId, s1[i]);
        }

        return retval;
    }

    double BayesianNetwork::getTransitionProbability(const Factors & space, const PartialFactors & s, const PartialFactors & s1) const {
        double retval = 1.0;
        // The matrix is made up of one component per child, and we
        // need to multiply all of them together. At each iteration we
        // look at a different "child".
        for (size_t j = 0; j < s1.first.size(); ++j) {
            // Find the matrix relative to this child
            const auto & node = nodes[s1.first[j]];
            // Compute the "dense" id for the needed parents
            // from the current domain.
            auto id = toIndexPartial(node.tag, space, s);
            // Multiply the current value by the lhs value.
            retval *= node.matrix(id, s1.second[j]);
        }
        return retval;
    }

    BasisFunction backProject(const Factors & space, const BayesianNetwork & b, const BasisFunction & bf) {
        // Here we have the two function inputs, in this form:
        //
        //     lhs: [parents, child] -> value
        //     rhs: [children] -> value
        BasisFunction retval;

        // The domain here depends on the parents of all elements of
        // the domain of the input basis.
        for (auto d : bf.tag)
            retval.tag = merge(retval.tag, b.nodes[d].tag);

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

    FactoredVector backProject(const Factors & space, const BayesianNetwork & b, const FactoredVector & fv) {
        FactoredVector retval;
        retval.bases.reserve(fv.bases.size());

        for (const auto & basis : fv.bases) {
            plusEqual(space, retval,
                    backProject(space, b, basis));
        }

        return retval;
    }
}
