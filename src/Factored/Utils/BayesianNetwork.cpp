#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    namespace Impl {
        template <typename DBN>
        double getTransitionProbabilityDBN(const DBN & dbn, const Factors & space, const Factors & s, const Factors & s1) {
            double retval = 1.0;

            // For each partial transition matrix, we compute the entry which
            // applies to this transition, and we multiply all entries together.
            for (size_t i = 0; i < space.size(); ++i) {
                // Compute parent ID based on the parents of state factor 'i'
                const auto parentId = toIndexPartial(dbn[i].tag, space, s);
                retval *= dbn[i].matrix(parentId, s1[i]);
            }

            return retval;
        }

        template <typename DBN>
        double getTransitionProbabilityDBN(const DBN & dbn, const Factors & space, const PartialFactors & s, const PartialFactors & s1) {
            double retval = 1.0;
            // The matrix is made up of one component per child, and we
            // need to multiply all of them together. At each iteration we
            // look at a different "child".
            for (size_t j = 0; j < s1.first.size(); ++j) {
                // Find the matrix relative to this child
                const auto & node = dbn[s1.first[j]];
                // Compute the "dense" id for the needed parents
                // from the current domain.
                const auto id = toIndexPartial(node.tag, space, s);
                // Multiply the current value by the lhs value.
                retval *= node.matrix(id, s1.second[j]);
            }
            return retval;
        }
    }

    // DBN

    double DBN::getTransitionProbability(const Factors & space, const Factors & s, const Factors & s1) const {
        return Impl::getTransitionProbabilityDBN(*this, space, s, s1);
    }

    double DBN::getTransitionProbability(const Factors & space, const PartialFactors & s, const PartialFactors & s1) const {
        return Impl::getTransitionProbabilityDBN(*this, space, s, s1);
    }

    const DBN::Node & DBN::operator[](size_t i) const {
        return nodes[i];
    }

    // DBNRef

    double DBNRef::getTransitionProbability(const Factors & space, const Factors & s, const Factors & s1) const {
        return Impl::getTransitionProbabilityDBN(*this, space, s, s1);
    }

    double DBNRef::getTransitionProbability(const Factors & space, const PartialFactors & s, const PartialFactors & s1) const {
        return Impl::getTransitionProbabilityDBN(*this, space, s, s1);
    }

    const DBN::Node & DBNRef::operator[](size_t i) const {
        return nodes[i].get();
    }

    // CompactDDN

    CompactDDN::CompactDynamicDecisionNetwork(
                std::vector<std::vector<Node>> diffs,
                DynamicBayesianNetwork defaultTransition
            ) : diffs_(std::move(diffs)), defaultTransition_(std::move(defaultTransition)) {}

    DBNRef CompactDDN::makeDiffTransition(const size_t a) const {
        DBNRef retval;
        retval.nodes.reserve(defaultTransition_.nodes.size());

        size_t j = 0;
        for (size_t i = 0; i < defaultTransition_.nodes.size(); ++i) {
            if (j < diffs_[a].size() && diffs_[a][j].id == i) {
                retval.nodes.emplace_back(std::ref(diffs_[a][j].node));
                ++j;
            } else {
                retval.nodes.emplace_back(std::ref(defaultTransition_.nodes[i]));
            }
        }
        return retval;
    }

    const DBN & CompactDDN::getDefaultTransition() const {
        return defaultTransition_;
    }

    const std::vector<std::vector<CompactDDN::Node>> & CompactDDN::getDiffNodes() const {
        return diffs_;
    }

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

    namespace Impl {
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
            PartialFactorsEnumerator domain(space, retval.tag);
            PartialFactorsEnumerator rhsDomain(space, bf.tag);
            for (size_t id = 0; domain.isValid(); domain.advance(), ++id) {
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
                for (size_t i = 0; rhsDomain.isValid(); rhsDomain.advance(), ++i)
                    currentVal += bf.values[i] * dbn.getTransitionProbability(space, *domain, *rhsDomain);
                rhsDomain.reset();

                retval.values[id] = currentVal;
            }
            return retval;
        }

        template <typename Net>
        FactoredVector backProject(const Factors & space, const Net & dbn, const FactoredVector & fv) {
            FactoredVector retval;
            retval.bases.reserve(fv.bases.size());

            for (const auto & basis : fv.bases) {
                // Note that we don't do plusEqual since we don't necessarily
                // want to merge entries here.
                retval.bases.emplace_back(backProject(space, dbn, basis));
            }

            return retval;
        }
    }

    BasisFunction backProject(const Factors & space, const DBN & dbn, const BasisFunction & bf) {
        return Impl::backProject(space, dbn, bf);
    }
    BasisFunction backProject(const Factors & space, const DBNRef & dbn, const BasisFunction & bf) {
        return Impl::backProject(space, dbn, bf);
    }
    FactoredVector backProject(const Factors & space, const DBN & dbn, const FactoredVector & fv) {
        return Impl::backProject(space, dbn, fv);
    }
    FactoredVector backProject(const Factors & space, const DBNRef & dbn, const FactoredVector & fv) {
        return Impl::backProject(space, dbn, fv);
    }

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
