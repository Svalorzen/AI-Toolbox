#ifndef AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE
#define AI_TOOLBOX_FACTORED_UTILS_BAYESIAN_NETWORK_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>

namespace AIToolbox::Factored {
    class DDNGraph {
        public:
            struct Node {
                PartialKeys agents;
                std::vector<PartialKeys> parents;
            };

            DDNGraph(State SS, Action AA) : S(std::move(SS)), A(std::move(AA)) {
                nodes_.reserve(S.size());
            }

            void pushNode(Node && node) {
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

            size_t getId(const State & s, const Action & a, const size_t feature) const {
                const auto actionId = toIndexPartial(nodes_[feature].agents, A, a);
                const auto & parents = nodes_[feature].parents[actionId];
                const auto parentId = toIndexPartial(parents, S, s);

                return startIds_[feature][actionId] + parentId;
            }

            size_t getId(const PartialState & s, const PartialAction & a, const size_t feature) const {
                const auto actionId = toIndexPartial(nodes_[feature].agents, A, a);
                const auto & parents = nodes_[feature].parents[actionId];
                const auto parentId = toIndexPartial(parents, S, s);

                return startIds_[feature][actionId] + parentId;
            }

            size_t getId(size_t actionId, size_t parentId, const size_t feature) const {
                return startIds_[feature][actionId] + parentId;
            }

            std::pair<size_t, size_t> getIds(const State & s, const Action & a, const size_t feature) const {
                const auto actionId = toIndexPartial(nodes_[feature].agents, A, a);
                const auto & parents = nodes_[feature].parents[actionId];
                const auto parentId = toIndexPartial(parents, S, s);

                return {actionId, parentId};
            }

            std::pair<size_t, size_t> getIds(const PartialState & s, const PartialAction & a, const size_t feature) const {
                const auto actionId = toIndexPartial(nodes_[feature].agents, A, a);
                const auto & parents = nodes_[feature].parents[actionId];
                const auto parentId = toIndexPartial(parents, S, s);

                return {actionId, parentId};
            }

            std::pair<size_t, size_t> getIds(size_t j, size_t feature) {
                // Start from the end (the -2 is there because the last element is the overall bound).
                std::pair<size_t, size_t> retval{startIds_[feature].size() - 2, 0};
                auto & [actionId, parentId] = retval;

                // While we are above, go down. This cannot go lower than zero,
                // so we only have to do 1 check.
                while (startIds_[feature][actionId] > j)
                    --actionId;

                parentId = j - startIds_[feature][actionId];

                return retval;
            }

            const State & getS() const { return S; }
            const Action & getA() const { return A; }
            const std::vector<Node> & getNodes() const { return nodes_; }
            size_t getSize(const size_t feature) const { return startIds_[feature].back(); }
            size_t getSize(const size_t actionId, const size_t feature) const {
                return startIds_[feature][actionId+1] - startIds_[feature][actionId];
            }

        private:
            State S;
            Action A;
            std::vector<Node> nodes_;
            std::vector<std::vector<size_t>> startIds_;
    };

    /**
     * @brief This class represents a Dynamic Decision Network with factored actions.
     *
     * This class is able to represent a Dynamic Decision Network with factored
     * actions, where the parents of each factor of the state depend on a
     * particular subset of actions.
     */
    struct FactoredDynamicDecisionNetwork {
        using TransitionMatrix = std::vector<Matrix2D>;

        /**
         * @brief This function returns the probability of a transition from one state to another with the given action.
         *
         * @param s The initial factors to start with.
         * @param a The selected action for the transition.
         * @param s1 The factors we should end up with.
         *
         * @return The probability of the transition.
         */
        double getTransitionProbability(const State & s, const Action & a, const State & s1) const;

        /**
         * @brief This function returns the probability of a transition from one state to another.
         *
         * This function allows to compute probabilities for subsets of
         * factors. The parent factors MUST contain all parents of the
         * children!
         *
         * @param s The initial factors to start with.
         * @param a The selected action for the transition.
         * @param s1 The factors we should end up with.
         *
         * @return The probability of the transition.
         */
        double getTransitionProbability(const PartialState & s, const PartialAction & a, const PartialState & s1) const;

        const DDNGraph & graph;
        TransitionMatrix transitions;
    };

    using FactoredDDN = FactoredDynamicDecisionNetwork;

    BasisMatrix backProject(const FactoredDDN & ddn, const BasisFunction & bf);
    FactoredMatrix2D backProject(const FactoredDDN & ddn, const FactoredVector & fv);
}

#endif
