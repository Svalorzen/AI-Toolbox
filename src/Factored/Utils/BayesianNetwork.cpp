#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    // FactoredDDN

    double FactoredDDN::getTransitionProbability(const Factors & s, const Factors & a, const Factors & s1) const {
        double retval = 1.0;

        // For each partial transition matrix, we compute the entry which
        // applies to this transition, and we multiply all entries together.
        for (size_t i = 0; i < graph.getS().size(); ++i) {
            retval *= transitions[i](graph.getId(s, a, i), s1[i]);
        }

        return retval;
    }

    double FactoredDDN::getTransitionProbability(const PartialFactors & s, const PartialFactors & a, const PartialFactors & s1) const {
        double retval = 1.0;

        // The matrix is made up of one component per child, and we
        // need to multiply all of them together. At each iteration we
        // look at a different "child".
        for (size_t j = 0; j < s1.first.size(); ++j) {
            const auto nodeId = s1.first[j];
            retval *= transitions[nodeId](graph.getId(s, a, nodeId), s1.second[j]);
        }

        return retval;
    }

    // Free functions

    BasisMatrix backProject(const FactoredDDN & ddn, const BasisFunction & rhs) {
        BasisMatrix retval;

        auto & graph = ddn.graph;
        auto & nodes = graph.getNodes();

        for (auto d : rhs.tag) {
            retval.actionTag = merge(retval.actionTag, nodes[d].agents);
            for (const auto & n : nodes[d].parents)
                retval.tag = merge(retval.tag, n);
        }

        const size_t sizeA = factorSpacePartial(retval.actionTag, graph.getA());
        const size_t sizeS = factorSpacePartial(retval.tag, graph.getS());

        retval.values.resize(sizeS, sizeA);

        PartialFactorsEnumerator sDomain(graph.getS(), retval.tag);
        PartialFactorsEnumerator aDomain(graph.getA(), retval.actionTag);
        PartialFactorsEnumerator rDomain(graph.getS(), rhs.tag);

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
                    currentVal += rhs.values[rId] * ddn.getTransitionProbability(*sDomain, *aDomain, *rDomain);
                rDomain.reset();

                retval.values(sId, aId) = currentVal;
            }
            aDomain.reset();
        }
        return retval;
    }

    FactoredMatrix2D backProject(const FactoredDDN & ddn, const FactoredVector & fv) {
        FactoredMatrix2D retval;
        retval.bases.reserve(fv.bases.size());

        for (const auto & basis : fv.bases) {
            // Note that we don't do plusEqual since we don't necessarily
            // want to merge entries here.
            retval.bases.emplace_back(backProject(ddn, basis));
        }

        return retval;
    }
}
