#ifndef AI_TOOLBOX_FACTOR_GRAPH_HEADER_FILE
#define AI_TOOLBOX_FACTOR_GRAPH_HEADER_FILE

#include <cstddef>
#include <list>
#include <vector>

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
     * This class maintains a single factor for any unique combination of
     * variables. When multiple factors are needed, a single Factor containing
     * a vector of data should suffice.
     *
     * @tparam Factor The Factor class that is stored for each factor.
     */
    template <typename Factor>
    class FactorGraph {
        public:
            struct VariableNode;

            using VariableList = std::vector<VariableNode>;
            using Variables    = std::vector<size_t>;

            class FactorNode {
                friend class FactorGraph;

                Factor f_;
                Variables variables_;

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

            struct VariableNode {
                FactorItList factors_;
            };

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
             * This function is very fast as the factors are kept in a
             * list, so removal is O(1).
             *
             * @param it An iterator to the factor to be removed.
             */
            void erase(FactorIt it);

            /**
             * @brief This function partially removes an variable from the graph.
             *
             * This function does not actually do much, so it is very
             * important that it is used correctly. No factors are
             * modified, so before calling this function all factors
             * pointing to this variable should be removed.
             *
             * This function simply clears the adjacency list for the
             * specified variable, and decreases the number of variables by one,
             * so that variableSize() reports one less variable.
             *
             * Calling this function multiple times will continue to
             * decrease the counter! This will have no side effect aside
             * that the variableSize() function will become meaningless.
             *
             * @param variable The variable to be removed.
             */
            void erase(size_t variable);

            /**
             * @brief This function returns the number of variables still in the graph.
             *
             * This function basically returns the number of variables this
             * graph has been initialized with, minus the number of times
             * the erase(size_t) function has been called (even for the
             * same variable!).
             *
             * This means that if the erase(size_t) function is used
             * generously, multiple times per variable, this function's return
             * value is meaningless.
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

            VariableList variableAdjacencies_;
            size_t activeVariables_;
    };

    template <typename Factor>
    FactorGraph<Factor>::FactorGraph(size_t variables) : variableAdjacencies_(variables), activeVariables_(variables) {}

    template <typename Factor>
    const typename FactorGraph<Factor>::FactorItList & FactorGraph<Factor>::getNeighbors(const size_t variable) const {
        return variableAdjacencies_[variable].factors_;
    }

    template <typename Factor>
    const typename FactorGraph<Factor>::Variables & FactorGraph<Factor>::getNeighbors(FactorIt factor) const {
        return factor->variables_;
    }

    template <typename Factor>
    const typename FactorGraph<Factor>::Variables & FactorGraph<Factor>::getNeighbors(CFactorIt factor) const {
        return factor->variables_;
    }

    template <typename Factor>
    typename FactorGraph<Factor>::Variables FactorGraph<Factor>::getNeighbors(const FactorItList & factors) const {
        Variables list1, list2;
        Variables *a1 = &list1, *a2 = &list2;
        for (const auto factor : factors) {
            const auto & variables = factor->variables_;
            std::set_union(std::begin(variables), std::end(variables), std::begin(*a1), std::end(*a1), std::back_inserter(*a2));
            std::swap(a1, a2);
            a2->clear();
        }
        return *a1;
    }

    template <typename Factor>
    typename FactorGraph<Factor>::FactorIt FactorGraph<Factor>::getFactor(const Variables & variables) {
        const auto found = factorByVariables_.find(variables);
        if (found != factorByVariables_.end())
            return found->second;

        factorAdjacencies_.emplace_back(FactorNode());
        auto it = --factorAdjacencies_.end();

        it->variables_ = variables;
        for (const auto a : variables)
            variableAdjacencies_[a].factors_.push_back(it);

        factorByVariables_[variables] = it;
        return it;
    }

    template <typename Factor>
    void FactorGraph<Factor>::erase(FactorIt it) {
        for (const auto variable : it->variables_) {
            auto & factors = variableAdjacencies_[variable].factors_;
            const auto foundIt = std::find(std::begin(factors), std::end(factors), it);
            if (foundIt != std::end(factors)) factors.erase(foundIt);
        }
        factorByVariables_.erase(it->variables_);
        factorAdjacencies_.erase(it);
    }

    template <typename Factor>
    void FactorGraph<Factor>::erase(const size_t a) {
        variableAdjacencies_[a].factors_.clear();
        --activeVariables_;
    }

    template <typename Factor>
    size_t FactorGraph<Factor>::variableSize() const  { return activeVariables_; }
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
    template <typename Factor>
    typename FactorGraph<Factor>::CFactorIt FactorGraph<Factor>::cbegin() const { return std::begin(factorAdjacencies_); }
    template <typename Factor>
    typename FactorGraph<Factor>::CFactorIt FactorGraph<Factor>::cend() const { return std::end(factorAdjacencies_); }
}

#endif
