#ifndef AI_TOOLBOX_FACTORED_FACTOR_GRAPH_HEADER_FILE
#define AI_TOOLBOX_FACTORED_FACTOR_GRAPH_HEADER_FILE

#include <cstddef>
#include <list>
#include <vector>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Types.hpp>
#include <boost/functional/hash.hpp>
#include <unordered_map>

namespace AIToolbox::Factored {
    /**
     * @brief This class offers a minimal interface to manager a factor graph.
     *
     * This class allows to store arbitrary data into each factor, and to
     * maintain adjacency lists between the factors and a given number of
     * variables. The interface is intentionally very simple and tries to do
     * very little, in order to allow clients to optimize their use of the
     * graph as much as possible.
     *
     * This class maintains a single FactorNode for any unique combination of
     * variables. When multiple factors are needed, a single FactorNode
     * containing a vector of data should suffice.
     *
     * @tparam FactorData The class that is stored for each FactorNode.
     */
    template <typename FactorData>
    class FactorGraph {
        public:
            using Variables    = PartialKeys;

            class FactorNode {
                friend class FactorGraph;

                FactorData f_;
                Variables variables_;

                public:
                    const Variables & getVariables() const { return variables_; }
                    const FactorData & getData() const { return f_; }
                    FactorData & getData() { return f_; }
            };

            using FactorList = std::list<FactorNode>;
            using FactorIt = typename FactorList::iterator;
            using CFactorIt = typename FactorList::const_iterator;
            using FactorItList = std::vector<FactorIt>;

            using value_type = FactorData;
            using iterator = FactorIt;
            using const_iterator = CFactorIt;

            /**
             * @brief Basic constructor.
             *
             * This constructor initializes the variable adjacency list with
             * the given number of variables. Variables in this class cannot be
             * added, only removed.
             *
             * @param variables The number of variables with which to start the graph.
             */
            FactorGraph(size_t variables);

            /**
             * @brief This function returns all factors adjacent to the given variable.
             *
             * @param variable The variable to look for.
             *
             * @return A list of iterators pointing at the factors adjacent to the given variable.
             */
            const FactorItList & getNeighbors(size_t variable) const;

            /**
             * @brief This function returns all variables adjacent to the given factor.
             *
             * @param factor An iterator to the factor to look for.
             *
             * @return A vector of variables adjacent to the given factor.
             */
            const Variables & getNeighbors(FactorIt factor) const;

            /**
             * @brief This function returns all variables adjacent to the given factor.
             *
             * @param factor A const iterator to the factor to look for.
             *
             * @return A vector of variables adjacent to the given factor.
             */
            const Variables & getNeighbors(CFactorIt factor) const;

            /**
             * @brief This function returns all variables adjacent to any of the given factors.
             *
             * This function is equivalent to calling the
             * getNeighbors(FactorIt) function multiple times, and merging
             * the results to eliminate duplicates.
             *
             * @param factors A list of iterators to the factors to look for.
             *
             * @return A vector of variables adjacent to any of the given factors.
             */
            Variables getNeighbors(const FactorItList & factors) const;

            /**
             * @brief This function returns an iterator to a factor adjacent to the given variables.
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
             * @param variables The variables the factor returned should be adjacent of.
             *
             * @return An iterator to the desired factor.
             */
            FactorIt getFactor(const Variables & variables);

            /**
             * @brief This function removes a factor from the graph.
             *
             * This function is very fast as the factors are kept in a list, so
             * that removal is O(1). We also need to remove the factor from
             * each agent's list, and that takes some more time.
             *
             * @param it An iterator to the factor to be removed.
             */
            void erase(FactorIt it);

            /**
             * @brief This function partially removes an variable from the graph.
             *
             * This function removes the selected variable, and ALL factors
             * associated with it.
             *
             * Removing the same variable more than once does not do anything.
             *
             * @param variable The variable to be removed.
             */
            void erase(size_t variable);

            /**
             * @brief This function returns the number of variables still in the graph.
             *
             * This function returns the number of active variables in the
             * graph (that have not been explicitly eliminated via
             * erase(size_t)).
             *
             * @return The number of variables still in the graph.
             */
            size_t variableSize() const;

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
             * @brief This function provides a const iterator to the beginning of the internal factor list.
             *
             * @return A const iterator to the beginning of the internal factor range.
             */
            CFactorIt cbegin() const;

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

            /**
             * @brief This function provides a const interactor to the end of the internal factor list.
             *
             * \sa factorsBegin() const
             *
             * @return A const iterator to the end of the internal factor range.
             */
            CFactorIt cend() const;

        private:
            FactorList factorAdjacencies_;
            std::unordered_map<Variables, FactorIt, boost::hash<Variables>> factorByVariables_;

            struct VariableNode {
                FactorItList factors;
                bool active = true;
            };

            std::vector<VariableNode> variableAdjacencies_;
            size_t activeVariables_;
    };

    template <typename FD>
    FactorGraph<FD>::FactorGraph(size_t variables) : variableAdjacencies_(variables), activeVariables_(variables) {}

    template <typename FD>
    const typename FactorGraph<FD>::FactorItList & FactorGraph<FD>::getNeighbors(const size_t variable) const {
        return variableAdjacencies_[variable].factors;
    }

    template <typename FD>
    const typename FactorGraph<FD>::Variables & FactorGraph<FD>::getNeighbors(FactorIt factor) const {
        return factor->variables_;
    }

    template <typename FD>
    const typename FactorGraph<FD>::Variables & FactorGraph<FD>::getNeighbors(CFactorIt factor) const {
        return factor->variables_;
    }

    template <typename FD>
    typename FactorGraph<FD>::Variables FactorGraph<FD>::getNeighbors(const FactorItList & factors) const {
        Variables retval;
        for (const auto factor : factors)
            set_union_inplace(retval, factor->variables_);

        return retval;
    }

    template <typename FD>
    typename FactorGraph<FD>::FactorIt FactorGraph<FD>::getFactor(const Variables & variables) {
        const auto found = factorByVariables_.find(variables);
        if (found != factorByVariables_.end())
            return found->second;

        factorAdjacencies_.emplace_back(FactorNode());
        auto it = --factorAdjacencies_.end();

        it->variables_ = variables;
        for (const auto a : variables)
            variableAdjacencies_[a].factors.push_back(it);

        factorByVariables_[variables] = it;
        return it;
    }

    template <typename FD>
    void FactorGraph<FD>::erase(FactorIt it) {
        for (const auto variable : it->variables_) {
            auto & factors = variableAdjacencies_[variable].factors;
            const auto foundIt = std::find(std::begin(factors), std::end(factors), it);
            if (foundIt != std::end(factors)) factors.erase(foundIt);
        }
        factorByVariables_.erase(it->variables_);
        factorAdjacencies_.erase(it);
    }

    template <typename FD>
    void FactorGraph<FD>::erase(const size_t a) {
        if (variableAdjacencies_[a].active) {
            for (auto it : variableAdjacencies_[a].factors) {
                for (const auto variable : it->variables_) {
                    if (variable == a) continue;
                    auto & factors = variableAdjacencies_[variable].factors;
                    const auto foundIt = std::find(std::begin(factors), std::end(factors), it);
                    if (foundIt != std::end(factors)) factors.erase(foundIt);
                }
                factorByVariables_.erase(it->variables_);
                factorAdjacencies_.erase(it);
            }
            variableAdjacencies_[a].factors.clear();
            variableAdjacencies_[a].active = false;
            --activeVariables_;
        }
    }

    template <typename FD>
    size_t FactorGraph<FD>::variableSize() const  { return activeVariables_; }
    template <typename FD>
    size_t FactorGraph<FD>::factorSize() const { return factorAdjacencies_.size(); }
    template <typename FD>
    typename FactorGraph<FD>::FactorIt FactorGraph<FD>::begin() { return std::begin(factorAdjacencies_); }
    template <typename FD>
    typename FactorGraph<FD>::FactorIt FactorGraph<FD>::end() { return std::end(factorAdjacencies_); }
    template <typename FD>
    typename FactorGraph<FD>::CFactorIt FactorGraph<FD>::begin() const { return std::begin(factorAdjacencies_); }
    template <typename FD>
    typename FactorGraph<FD>::CFactorIt FactorGraph<FD>::end() const { return std::end(factorAdjacencies_); }
    template <typename FD>
    typename FactorGraph<FD>::CFactorIt FactorGraph<FD>::cbegin() const { return std::begin(factorAdjacencies_); }
    template <typename FD>
    typename FactorGraph<FD>::CFactorIt FactorGraph<FD>::cend() const { return std::end(factorAdjacencies_); }
}

#endif
