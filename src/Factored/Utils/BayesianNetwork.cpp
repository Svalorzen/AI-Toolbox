#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    // DDNGraph

    DDNGraph::DynamicDecisionNetworkGraph(State SS, Action AA) : S(std::move(SS)), A(std::move(AA)) {
        nodes_.reserve(S.size());
    }

    void DDNGraph::pushNode(const Node & node) {
        auto n = node;
        pushNode(std::move(n));
    }

    void DDNGraph::pushNode(Node && node) {
        // Begin sanity check to only construct graphs that make sense.
        if (nodes_.size() == S.size())
            throw std::runtime_error("Pushed too many nodes in DDNGraph");

        TagErrors error;
        std::tie(error, std::ignore) = checkTag(A, node.agents);
        switch (error) {
            case TagErrors::NoElements:
                throw std::invalid_argument("Pushed node in DDNGraph contains agents tag with no elements!");
            case TagErrors::TooManyElements:
                throw std::invalid_argument("Pushed node in DDNGraph contains agents tag with too many elements!");
            case TagErrors::IdTooHigh:
                throw std::invalid_argument("Pushed node in DDNGraph references agent IDs too high for the action space!");
            case TagErrors::NotSorted:
                throw std::invalid_argument("Pushed node in DDNGraph contains agents tag that are not sorted!");
            case TagErrors::Duplicates:
                throw std::invalid_argument("Pushed node in DDNGraph contains duplicate agents in agents tag!");
            default:;
        }

        if (node.parents.size() != factorSpacePartial(node.agents, A))
            throw std::invalid_argument("Pushed node DDNGraph has an incorrect number of parent sets for the specified agents tag!");

        for (size_t i = 0; i < node.parents.size(); ++i) {
            std::tie(error, std::ignore) = checkTag(S, node.parents[i]);

            switch (error) {
                case TagErrors::NoElements:
                    throw std::invalid_argument("Pushed node in DDNGraph contains parents tags with no elements!");
                case TagErrors::TooManyElements:
                    throw std::invalid_argument("Pushed node in DDNGraph contains parents tags with too many elements!");
                case TagErrors::IdTooHigh:
                    throw std::invalid_argument("Pushed node in DDNGraph references parent IDs too high for the state space!");
                case TagErrors::NotSorted:
                    throw std::invalid_argument("Pushed node in DDNGraph contains parents tags that are not sorted!");
                case TagErrors::Duplicates:
                    throw std::invalid_argument("Pushed node in DDNGraph contains duplicate parents in parents tags!");
                default:;
            }
        }

        // Sanity check ended, we can pull the node in.
        nodes_.emplace_back(std::move(node));

        auto & newNode = nodes_.back();
        startIds_.emplace_back(newNode.parents.size() + 1);
        auto & newStartIds = startIds_.back();

        size_t newStartId = 0;
        for (size_t i = 0; i < newNode.parents.size(); ++i) {
            newStartIds[i] = newStartId;
            newStartId += factorSpacePartial(newNode.parents[i], S);
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
        const auto actionId = toIndexPartial(nodes_[feature].agents, A, a);
        const auto & parents = nodes_[feature].parents[actionId];
        const auto parentId = toIndexPartial(parents, S, s);

        return {parentId, actionId};
    }

    std::pair<size_t, size_t> DDNGraph::getIds(const size_t feature, const PartialState & s, const PartialAction & a) const {
        const auto actionId = toIndexPartial(nodes_[feature].agents, A, a);
        const auto & parents = nodes_[feature].parents[actionId];
        const auto parentId = toIndexPartial(parents, S, s);

        return {parentId, actionId};
    }

    std::pair<size_t, size_t> DDNGraph::getIds(const size_t feature, const size_t j) {
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
        return nodes_[feature].parents.size();
    }
    size_t DDNGraph::getPartialSize(const size_t feature, const size_t actionId) const {
        return startIds_[feature][actionId+1] - startIds_[feature][actionId];
    }
    const State & DDNGraph::getS() const { return S; }
    const Action & DDNGraph::getA() const { return A; }
    const std::vector<DDNGraph::Node> & DDNGraph::getNodes() const { return nodes_; }

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
