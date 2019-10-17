#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    // FactoredDDN

    double FactoredDDN::getTransitionProbability(const Factors & space, const Factors & actions, const Factors & s, const Factors & a, const Factors & s1) const {
        double retval = 1.0;

        // For each partial transition matrix, we compute the entry which
        // applies to this transition, and we multiply all entries together.
        for (size_t i = 0; i < space.size(); ++i) {
            const auto & node = nodes[i];
            // Compute action ID based on the actions that affect state factor 'i'.
            const auto actionId = toIndexPartial(node.actionTag, actions, a);
            // Compute parent ID based on the parents of state factor 'i' under this action.
            const auto parentId = toIndexPartial(node.nodes[actionId].tag, space, s);

            retval *= node.nodes[actionId].matrix(parentId, s1[i]);
        }

        return retval;
    }

    double FactoredDDN::getTransitionProbability(const Factors & space, const Factors & actions, const PartialFactors & s, const PartialFactors & a, const PartialFactors & s1) const {
        double retval = 1.0;

        // The matrix is made up of one component per child, and we
        // need to multiply all of them together. At each iteration we
        // look at a different "child".
        for (size_t j = 0; j < s1.first.size(); ++j) {
            const auto & node = nodes[s1.first[j]];
            // Compute action ID based on the actions that affect state factor 'i'.
            const auto actionId = toIndexPartial(node.actionTag, actions, a);
            // Compute parent ID based on the parents of state factor 'i' under this action.
            const auto parentId = toIndexPartial(node.nodes[actionId].tag, space, s);

            retval *= node.nodes[actionId].matrix(parentId, s1.second[j]);
        }

        return retval;
    }

    const FactoredDDN::Node & FactoredDDN::operator[](size_t i) const {
        return nodes[i];
    }

    // Free functions

    BasisMatrix backProject(const Factors & space, const Factors & actions, const FactoredDDN & ddn, const BasisFunction & rhs) {
        BasisMatrix retval;

        for (auto d : rhs.tag) {
            retval.actionTag = merge(retval.actionTag, ddn[d].actionTag);
            for (const auto & n : ddn[d].nodes)
                retval.tag = merge(retval.tag, n.tag);
        }

        const size_t sizeA = factorSpacePartial(retval.actionTag, actions);
        const size_t sizeS = factorSpacePartial(retval.tag, space);

        retval.values.resize(sizeS, sizeA);

        PartialFactorsEnumerator sDomain(space, retval.tag);
        PartialFactorsEnumerator aDomain(actions, retval.actionTag);
        PartialFactorsEnumerator rDomain(space, rhs.tag);

        for (size_t sId = 0; sDomain.isValid(); sDomain.advance(), ++sId) {
            for (size_t aId = 0; aDomain.isValid(); aDomain.advance(), ++aId) {
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
                for (size_t rId = 0; rDomain.isValid(); rDomain.advance(), ++rId)
                    currentVal += rhs.values[rId] * ddn.getTransitionProbability(space, actions, *sDomain, *aDomain, *rDomain);
                rDomain.reset();

                retval.values(sId, aId) = currentVal;
            }
            aDomain.reset();
        }
        return retval;
    }

    FactoredMatrix2D backProject(const Factors & space, const Factors & actions, const FactoredDDN & ddn, const FactoredVector & fv) {
        FactoredMatrix2D retval;
        retval.bases.reserve(fv.bases.size());

        for (const auto & basis : fv.bases) {
            // Note that we don't do plusEqual since we don't necessarily
            // want to merge entries here.
            retval.bases.emplace_back(backProject(space, actions, ddn, basis));
        }

        return retval;
    }
}
