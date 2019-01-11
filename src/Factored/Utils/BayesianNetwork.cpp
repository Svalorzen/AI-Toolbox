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
                auto parentId = toIndexPartial(dbn[i].tag, space, s);
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
                auto id = toIndexPartial(node.tag, space, s);
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

    // --------------

    CompactDDN::CompactDynamicDecisionNetwork(
                std::vector<std::vector<Node>> diffs,
                DynamicBayesianNetwork defaultTransition
            ) : diffs_(std::move(diffs)), defaultTransition_(std::move(defaultTransition)) {}

    DBNRef CompactDDN::makeDiffTransition(const size_t a) const {
        DBNRef retval;
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
}
