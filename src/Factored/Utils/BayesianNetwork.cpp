#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    // DDNGraph

    DDNGraph::DynamicDecisionNetworkGraph(State SS, Action AA) : S(std::move(SS)), A(std::move(AA)) {
        parents_.reserve(S.size());
        startIds_.reserve(S.size());
    }

    void DDNGraph::push(ParentSet parents) {
        // Begin sanity check to only construct graphs that make sense.
        if (parents_.size() == S.size())
            throw std::runtime_error("Pushed too many parent sets in DDNGraph");

        TagErrors error;
        std::tie(error, std::ignore) = checkTag(A, parents.agents);
        switch (error) {
            case TagErrors::NoElements:
                throw std::invalid_argument("Pushed parent set in DDNGraph contains agents tag with no elements!");
            case TagErrors::TooManyElements:
                throw std::invalid_argument("Pushed parent set in DDNGraph contains agents tag with too many elements!");
            case TagErrors::IdTooHigh:
                throw std::invalid_argument("Pushed parent set in DDNGraph references agent IDs too high for the action space!");
            case TagErrors::NotSorted:
                throw std::invalid_argument("Pushed parent set in DDNGraph contains agents tag that are not sorted!");
            case TagErrors::Duplicates:
                throw std::invalid_argument("Pushed parent set in DDNGraph contains duplicate agents in agents tag!");
            default:;
        }

        if (parents.features.size() != factorSpacePartial(parents.agents, A))
            throw std::invalid_argument("Pushed parent set DDNGraph has an incorrect number of feature sets for the specified agents tag!");

        for (size_t i = 0; i < parents.features.size(); ++i) {
            std::tie(error, std::ignore) = checkTag(S, parents.features[i]);

            switch (error) {
                case TagErrors::NoElements:
                    throw std::invalid_argument("Pushed parent set in DDNGraph contains feature tags with no elements!");
                case TagErrors::TooManyElements:
                    throw std::invalid_argument("Pushed parent set in DDNGraph contains feature tags with too many elements!");
                case TagErrors::IdTooHigh:
                    throw std::invalid_argument("Pushed parent set in DDNGraph references parent IDs too high for the state space!");
                case TagErrors::NotSorted:
                    throw std::invalid_argument("Pushed parent set in DDNGraph contains feature tags that are not sorted!");
                case TagErrors::Duplicates:
                    throw std::invalid_argument("Pushed parent set in DDNGraph contains duplicate features in feature tags!");
                default:;
            }
        }

        // Sanity check ended, we can pull the node in.
        parents_.emplace_back(std::move(parents));

        auto & newParents = parents_.back();
        startIds_.emplace_back(newParents.features.size() + 1);
        auto & newStartIds = startIds_.back();

        size_t newStartId = 0;
        for (size_t i = 0; i < newParents.features.size(); ++i) {
            newStartIds[i] = newStartId;
            newStartId += factorSpacePartial(newParents.features[i], S);
        }
        // Save overall length needed to store one element per parent
        // set for this node.
        newStartIds.back() = newStartId;
    }

    // ID CODE
    //
    // Note that while our API always tries to keep arguments as
    // <state,action>, internally the action ID always goes first, as it
    // specifies which parent set to use.

    size_t DDNGraph::getId(const size_t feature, const State & s, const Action & a) const {
        const auto [parentId, actionId] = getIds(feature, s, a);

        return getId(feature, parentId, actionId);
    }

    size_t DDNGraph::getId(const size_t feature, const PartialState & s, const PartialAction & a) const {
        const auto [parentId, actionId] = getIds(feature, s, a);

        return getId(feature, parentId, actionId);
    }

    size_t DDNGraph::getId(const size_t feature, size_t parentId, size_t actionId) const {
        return startIds_[feature][actionId] + parentId;
    }

    std::pair<size_t, size_t> DDNGraph::getIds(const size_t feature, const State & s, const Action & a) const {
        const auto actionId = toIndexPartial(parents_[feature].agents, A, a);
        const auto & features = parents_[feature].features[actionId];
        const auto parentId = toIndexPartial(features, S, s);

        return {parentId, actionId};
    }

    std::pair<size_t, size_t> DDNGraph::getIds(const size_t feature, const PartialState & s, const PartialAction & a) const {
        const auto actionId = toIndexPartial(parents_[feature].agents, A, a);
        const auto & features = parents_[feature].features[actionId];
        const auto parentId = toIndexPartial(features, S, s);

        return {parentId, actionId};
    }

    std::pair<size_t, size_t> DDNGraph::getIds(const size_t feature, const size_t j) const {
        // Start from the end (the -2 is there because the last element is the overall bound).
        std::pair<size_t, size_t> retval{0, startIds_[feature].size() - 2};
        auto & [parentId, actionId] = retval;

        // While we are above, go down. This cannot go lower than zero,
        // so we only have to do 1 check.
        while (startIds_[feature][actionId] > j)
            --actionId;

        parentId = j - startIds_[feature][actionId];

        return retval;
    }

    size_t DDNGraph::getSize(const size_t feature) const {
        return startIds_[feature].back();
    }

    size_t DDNGraph::getPartialSize(const size_t feature) const {
        return parents_[feature].features.size();
    }
    size_t DDNGraph::getPartialSize(const size_t feature, const size_t actionId) const {
        return startIds_[feature][actionId+1] - startIds_[feature][actionId];
    }
    const State & DDNGraph::getS() const { return S; }
    const Action & DDNGraph::getA() const { return A; }
    const std::vector<DDNGraph::ParentSet> & DDNGraph::getParentSets() const { return parents_; }

    // DDN

    double DDN::getTransitionProbability(const Factors & s, const Factors & a, const Factors & s1) const {
        double retval = 1.0;

        // For each partial transition matrix, we compute the entry which
        // applies to this transition, and we multiply all entries together.
        for (size_t i = 0; i < graph.getS().size(); ++i) {
            retval *= transitions[i](graph.getId(i, s, a), s1[i]);
        }

        return retval;
    }

    double DDN::getTransitionProbability(const PartialFactors & s, const PartialFactors & a, const PartialFactors & s1) const {
        double retval = 1.0;

        // The matrix is made up of one component per child, and we
        // need to multiply all of them together. At each iteration we
        // look at a different "child".
        for (size_t j = 0; j < s1.first.size(); ++j) {
            const auto nodeId = s1.first[j];
            retval *= transitions[nodeId](graph.getId(nodeId, s, a), s1.second[j]);
        }

        return retval;
    }

    // Free functions

    BasisMatrix backProject(const DDN & ddn, const BasisFunction & rhs) {
        BasisMatrix retval;

        auto & graph = ddn.graph;
        auto & parentSets = graph.getParentSets();

        for (auto d : rhs.tag) {
            retval.actionTag = merge(retval.actionTag, parentSets[d].agents);
            for (const auto & n : parentSets[d].features)
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

    FactoredMatrix2D backProject(const DDN & ddn, const FactoredVector & fv) {
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
