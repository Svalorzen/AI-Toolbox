#ifndef AI_TOOLBOX_FACTOR_GRAPH_HEADER_FILE
#define AI_TOOLBOX_FACTOR_GRAPH_HEADER_FILE

#include <cstddef>
#include <list>
#include <vector>

#include <AIToolbox/FactoredMDP/Types.hpp>
#include <boost/functional/hash.hpp>
#include <unordered_map>

namespace AIToolbox::FactoredMDP {
    /**
     * @brief This class offers a minimal interface to manager a factor graph.
     *
     * This class allows to store arbitrary data into each factor, and to
     * maintain adjacency lists between the factors and a given number of
     * agents. The interface is intentionally very simple and tries to do
     * very little, in order to allow clients to optimize their use of the
     * graph as much as possible.
     *
     * This class maintains a single factor for any unique combination of
     * agents. When multiple factors are needed, a single Factor containing
     * a vector of data should suffice.
     *
     * @tparam Factor The Factor class that is stored for each factor.
     */
    template <typename Factor>
    class FactorGraph {
        public:
            struct AgentNode;

            using AgentList = std::vector<AgentNode>;
            using Agents    = std::vector<size_t>;

            class FactorNode {
                friend class FactorGraph;

                Factor f_;
                Agents agents_;

                public:
                    const Factor & getData() const { return f_; }
                    Factor & getData() { return f_; }
            };

            using FactorList = std::list<FactorNode>;
            using FactorIt = typename FactorList::iterator;
            using CFactorIt = typename FactorList::const_iterator;
            using FactorItList = std::vector<FactorIt>;

            using value_type = Factor;
            using iterator = FactorIt;
            using const_iterator = CFactorIt;

            struct AgentNode {
                FactorItList factors_;
            };

            /**
             * @brief Basic constructor.
             *
             * This constructor initializes the agent adjacency list with
             * the given number of agents. Agents in this class cannot be
             * added, only removed.
             *
             * @param agents The number of agents with which to start the graph.
             */
            FactorGraph(size_t agents);

            /**
             * @brief This function returns all factors adjacent to the given agent.
             *
             * @param agent The agent to look for.
             *
             * @return A list of iterators pointing at the factors adjacent to the given agent.
             */
            const FactorItList & getNeighbors(size_t agent) const;

            /**
             * @brief This function returns all agents adjacent to the given factor.
             *
             * @param factor An iterator to the factor to look for.
             *
             * @return A vector of agents adjacent to the given factor.
             */
            const Agents & getNeighbors(FactorIt factor) const;

            /**
             * @brief This function returns all agents adjacent to the given factor.
             *
             * @param factor A const iterator to the factor to look for.
             *
             * @return A vector of agents adjacent to the given factor.
             */
            const Agents & getNeighbors(CFactorIt factor) const;

            /**
             * @brief This function returns all agents adjacent to any of the given factors.
             *
             * This function is equivalent to calling the
             * getNeighbors(FactorIt) function multiple times, and merging
             * the results to eliminate duplicates.
             *
             * @param factors A list of iterators to the factors to look for.
             *
             * @return A vector of agents adjacent to any of the given factors.
             */
            Agents getNeighbors(const FactorItList & factors) const;

            /**
             * @brief This function returns an iterator to a factor adjacent to the given agents.
             *
             * This function may return an iterator to an already existing
             * factor, or if it didn't exist before, to a newly created
             * one.
             *
             * This means that it is safe to call this function multiple
             * times with the same input, as only one factor will be
             * created.
             *
             * As factors are kept in a list, insertion is O(1).
             *
             * @param agents The agents the factor returned should be adjacent of.
             *
             * @return An iterator to the desired factor.
             */
            FactorIt getFactor(const Agents & agents);

            /**
             * @brief This function removes a factor from the graph.
             *
             * This function is very fast as the factors are kept in a
             * list, so removal is O(1).
             *
             * @param it An iterator to the factor to be removed.
             */
            void erase(FactorIt it);

            /**
             * @brief This function partially removes an agent from the graph.
             *
             * This function does not actually do much, so it is very
             * important that it is used correctly. No factors are
             * modified, so before calling this function all factors
             * pointing to this agent should be removed.
             *
             * This function simply clears the adjacency list for the
             * specified agent, and decreases the number of agents by one,
             * so that agentSize() reports one less agent.
             *
             * Calling this function multiple times will continue to
             * decrease the counter! This will have no side effect aside
             * that the agentSize() function will become meaningless.
             *
             * @param agent The agent to be removed.
             */
            void erase(size_t agent);

            /**
             * @brief This function returns the number of agents still in the graph.
             *
             * This function basically returns the number of agents this
             * graph has been initialized with, minus the number of times
             * the erase(size_t) function has been called (even for the
             * same agent!).
             *
             * This means that if the erase(size_t) function is used
             * generously, multiple times per agent, this function's return
             * value is meaningless.
             *
             * @return The number of agents still in the graph.
             */
            size_t agentSize() const;

            /**
             * @brief This function returns the number of factors still in the graph.
             *
             * @return The number of factors still in the graph.
             */
            size_t factorSize() const;

            /**
             * @brief This function provides an editable iterator to the beginning of the internal factor list.
             *
             * This function is used in order to allow editing of all the factors.
             *
             * @return An iterator to the beginning of the internal factor range.
             */
            FactorIt begin();

            /**
             * @brief This function provides a const iterator to the beginning of the internal factor list.
             *
             * @return A const iterator to the beginning of the internal factor range.
             */
            CFactorIt begin() const;

            /**
             * @brief This function provides an editable interactor to the end of the internal factor list.
             *
             * \sa factorsBegin()
             *
             * @return An iterator to the end of the internal factor range.
             */
            FactorIt end();

            /**
             * @brief This function provides a const interactor to the end of the internal factor list.
             *
             * \sa factorsBegin() const
             *
             * @return A const iterator to the end of the internal factor range.
             */
            CFactorIt end() const;

        private:
            FactorList factorAdjacencies_;
            std::unordered_map<Agents, FactorIt, boost::hash<Agents>> factorByAgents_;

            AgentList agentAdjacencies_;
            size_t activeAgents_;
    };

    template <typename Factor>
    FactorGraph<Factor>::FactorGraph(size_t agents) : agentAdjacencies_(agents), activeAgents_(agents) {}

    template <typename Factor>
    const typename FactorGraph<Factor>::FactorItList & FactorGraph<Factor>::getNeighbors(const size_t agent) const {
        return agentAdjacencies_[agent].factors_;
    }

    template <typename Factor>
    const typename FactorGraph<Factor>::Agents & FactorGraph<Factor>::getNeighbors(FactorIt factor) const {
        return factor->agents_;
    }

    template <typename Factor>
    const typename FactorGraph<Factor>::Agents & FactorGraph<Factor>::getNeighbors(CFactorIt factor) const {
        return factor->agents_;
    }

    template <typename Factor>
    typename FactorGraph<Factor>::Agents FactorGraph<Factor>::getNeighbors(const FactorItList & factors) const {
        Agents list1, list2;
        Agents *a1 = &list1, *a2 = &list2;
        for (const auto factor : factors) {
            const auto & agents = factor->agents_;
            std::set_union(std::begin(agents), std::end(agents), std::begin(*a1), std::end(*a1), std::back_inserter(*a2));
            std::swap(a1, a2);
            a2->clear();
        }
        return *a1;
    }

    template <typename Factor>
    typename FactorGraph<Factor>::FactorIt FactorGraph<Factor>::getFactor(const Agents & agents) {
        const auto found = factorByAgents_.find(agents);
        if (found != factorByAgents_.end())
            return found->second;

        factorAdjacencies_.emplace_back(FactorNode());
        auto it = --factorAdjacencies_.end();

        it->agents_ = agents;
        for (const auto a : agents)
            agentAdjacencies_[a].factors_.push_back(it);

        factorByAgents_[agents] = it;
        return it;
    }

    template <typename Factor>
    void FactorGraph<Factor>::erase(FactorIt it) {
        for (const auto agent : it->agents_) {
            auto & factors = agentAdjacencies_[agent].factors_;
            const auto foundIt = std::find(std::begin(factors), std::end(factors), it);
            if (foundIt != std::end(factors)) factors.erase(foundIt);
        }
        factorByAgents_.erase(it->agents_);
        factorAdjacencies_.erase(it);
    }

    template <typename Factor>
    void FactorGraph<Factor>::erase(const size_t a) {
        agentAdjacencies_[a].factors_.clear();
        --activeAgents_;
    }

    template <typename Factor>
    size_t FactorGraph<Factor>::agentSize() const  { return activeAgents_; }
    template <typename Factor>
    size_t FactorGraph<Factor>::factorSize() const { return factorAdjacencies_.size(); }
    template <typename Factor>
    typename FactorGraph<Factor>::FactorIt FactorGraph<Factor>::begin() { return std::begin(factorAdjacencies_); }
    template <typename Factor>
    typename FactorGraph<Factor>::FactorIt FactorGraph<Factor>::end() { return std::end(factorAdjacencies_); }
    template <typename Factor>
    typename FactorGraph<Factor>::CFactorIt FactorGraph<Factor>::begin() const { return std::begin(factorAdjacencies_); }
    template <typename Factor>
    typename FactorGraph<Factor>::CFactorIt FactorGraph<Factor>::end() const { return std::end(factorAdjacencies_); }
}

#endif
