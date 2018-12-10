#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    ParametricBayesianNetwork::ParametricBayesianNetwork(
                std::vector<std::vector<BayesianDiffNode>> diffs,
                BayesianNetwork<false> defaultTransition
            ) : diffs_(std::move(diffs)), defaultTransition_(std::move(defaultTransition)) {}

    const BayesianNetwork<false> & ParametricBayesianNetwork::getDefaultTransition() const {
        return defaultTransition_;
    }

    BayesianNetwork<true> ParametricBayesianNetwork::makeDiffTransition(const size_t a) const {
        BayesianNetwork<true> retval;
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
}
