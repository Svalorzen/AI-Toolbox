#ifndef AI_TOOLBOX_FACTORED_FACTOR_GRAPH_HEADER_FILE
#define AI_TOOLBOX_FACTORED_FACTOR_GRAPH_HEADER_FILE

#include <cstddef>
#include <list>
#include <vector>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Types.hpp>

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
     * This class can allocate list nodes from an internal static pool; so it
     * is probably not thread-safe.
     *
     * @tparam FactorData The class that is stored for each FactorNode.
     */
    template <typename FactorData>
    class FactorGraph {
        public:
            using Variables = PartialKeys;

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
             * @brief Copy constructor.
             *
             * This constructor simply copies the graph, but also makes sure that the internal
             * stored iterators point to the correct containers.
             *
             * We do not check the internal FactorData, so make sure it doesn't
             * have pointers or iterators inside!
             *
             * @param other The graph to copy.
             */
            FactorGraph(const FactorGraph & other);

            /**
             * @brief Deleted assignment operator.
             *
             * Since we already provide a copy constructor, this is redundant.
             * The assignment operator provided by default would be broken,
             * so we remove it.
             */
            FactorGraph & operator=(const FactorGraph &) = delete;

            /**
             * @brief This function returns all factors adjacent to the given variable.
             *
             * @param variable The variable to look for.
             *
             * @return A list of iterators pointing at the factors adjacent to the given variable.
             */
            const FactorItList & getFactors(size_t variable) const;

            /**
             * @brief This function returns all variables adjacent to a factor adjacent to the input variable.
             *
             * @param variable The variable to look for.
             *
             * @return All *other* variables connected in some way to the input one.
             */
            const Variables & getVariables(size_t variable) const;

            /**
             * @brief This function returns all variables adjacent to the given factor.
             *
             * @param factor An iterator to the factor to look for.
             *
             * @return A vector of variables adjacent to the given factor.
             */
            const Variables & getVariables(FactorIt factor) const;

            /**
             * @brief This function returns all variables adjacent to the given factor.
             *
             * @param factor A const iterator to the factor to look for.
             *
             * @return A vector of variables adjacent to the given factor.
             */
            const Variables & getVariables(CFactorIt factor) const;

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
            Variables getVariables(const FactorItList & factors) const;

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

            /**
             * @brief This function returns the variable which is the cheapest to remove with GenericVariableElimination.
             *
             * The choice is made heuristically, as computing the true best is
             * an NP-Complete problem.
             *
             * @param F The value space for each variable.
             *
             * @return The best variable to eliminate.
             */
            size_t bestVariableToRemove(const Factors & F) const;

        private:
            FactorList factorAdjacencies_;
            static FactorList factorAdjacenciesPool_;

            auto findFactorByVariables(const FactorItList & list, const Variables & variables) const {
                return std::find_if(
                    std::begin(list),
                    std::end(list),
                    [&variables](const FactorIt it){ return it->variables_ == variables; }
                );
            }

            struct VariableNode {
                FactorItList factors;
                Variables vNeighbors;
                bool active = true;
            };

            std::vector<VariableNode> variableAdjacencies_;
            size_t activeVariables_;
    };

    template <typename FD>
    typename FactorGraph<FD>::FactorList FactorGraph<FD>::factorAdjacenciesPool_;

    template <typename FD>
    FactorGraph<FD>::FactorGraph(size_t variables) : variableAdjacencies_(variables), activeVariables_(variables) {}

    template <typename FD>
    FactorGraph<FD>::FactorGraph(const FactorGraph & other) :
        variableAdjacencies_(other.variableAdjacencies_),
        activeVariables_(other.activeVariables_)
    {
        // Try to take as much memory from the pool as possible.
        auto oIt = std::begin(other.factorAdjacencies_);
        while (factorAdjacencies_.size() < other.factorAdjacencies_.size()) {
            if (factorAdjacenciesPool_.size() > 0) {
                factorAdjacencies_.splice(std::end(factorAdjacencies_), factorAdjacenciesPool_, std::begin(factorAdjacenciesPool_));
                factorAdjacencies_.back() = *oIt++;
            } else {
                factorAdjacencies_.emplace_back(*oIt++);
            }
        }

        // So here it's pretty simple; we just need to rebuild the 'factors'
        // variable in each VariableNode, as it contains iterators that need to
        // point to the newly copied lists, rather than the ones in 'other'.

        // First we cleanup the old iterators.
        for (auto & a : variableAdjacencies_)
            a.factors.clear();

        // Then we rebuild them.
        for (auto it = factorAdjacencies_.begin(); it != factorAdjacencies_.end(); ++it) {
            for (auto a : it->getVariables()) {
                variableAdjacencies_[a].factors.push_back(it);
            }
        }
    }

    template <typename FD>
    const typename FactorGraph<FD>::FactorItList & FactorGraph<FD>::getFactors(const size_t variable) const {
        return variableAdjacencies_[variable].factors;
    }

    template <typename FD>
    const typename FactorGraph<FD>::Variables & FactorGraph<FD>::getVariables(const size_t variable) const {
        return variableAdjacencies_[variable].vNeighbors;
    }

    template <typename FD>
    const typename FactorGraph<FD>::Variables & FactorGraph<FD>::getVariables(FactorIt factor) const {
        return factor->variables_;
    }

    template <typename FD>
    const typename FactorGraph<FD>::Variables & FactorGraph<FD>::getVariables(CFactorIt factor) const {
        return factor->variables_;
    }

    template <typename FD>
    typename FactorGraph<FD>::Variables FactorGraph<FD>::getVariables(const FactorItList & factors) const {
        Variables retval;
        for (const auto factor : factors)
            set_union_inplace(retval, factor->variables_);

        return retval;
    }

    template <typename FD>
    typename FactorGraph<FD>::FactorIt FactorGraph<FD>::getFactor(const Variables & variables) {
        const auto found = findFactorByVariables(variableAdjacencies_[variables[0]].factors, variables);
        if (found != variableAdjacencies_[variables[0]].factors.end())
            return *found;

        FactorIt it;
        if (!factorAdjacenciesPool_.size()) {
            factorAdjacencies_.emplace_back(FactorNode());
            it = --factorAdjacencies_.end();
        } else {
            factorAdjacencies_.splice(std::begin(factorAdjacencies_), factorAdjacenciesPool_, std::begin(factorAdjacenciesPool_));
            it = factorAdjacencies_.begin();
            // We reset the data; just in case it's a vector we don't want to
            // move but assign so that it does not clear already allocated
            // memory.
            auto tmp = FD{};
            it->f_ = tmp;
        }

        it->variables_ = variables;
        for (const auto a : variables) {
            auto & va = variableAdjacencies_[a];
            va.factors.push_back(it);

            // Add *other* agents to vNeighbors
            const auto mid = va.vNeighbors.size();
            va.vNeighbors.reserve(mid + variables.size() - 1);

            for (size_t i = 0, j = 0; i < variables.size(); ) {
                if (variables[i] == a) {
                    ++i;
                } else if (j == mid || variables[i] < va.vNeighbors[j]) {
                    va.vNeighbors.push_back(variables[i]);
                    ++i;
                } else {
                    if (variables[i] == va.vNeighbors[j])
                        ++i;
                    ++j;
                }
            }
            std::inplace_merge(std::begin(va.vNeighbors), std::begin(va.vNeighbors)+mid, std::end(va.vNeighbors));
        }
        return it;
    }

    template <typename FD>
    void FactorGraph<FD>::erase(const size_t a) {
        auto & va = variableAdjacencies_[a];
        if (!va.active) return;

        for (auto it : va.factors) {
            for (const auto variable : it->variables_) {
                if (variable == a) continue;

                auto & factors = variableAdjacencies_[variable].factors;
                const auto foundIt = std::find(std::begin(factors), std::end(factors), it);

                assert(foundIt != std::end(factors));
                factors.erase(foundIt);
            }
            factorAdjacenciesPool_.splice(std::begin(factorAdjacenciesPool_), factorAdjacencies_, it);
        }
        for (const auto aa : va.vNeighbors) {
            auto & vaa = variableAdjacencies_[aa];
            vaa.vNeighbors.erase(std::find(std::begin(vaa.vNeighbors), std::end(vaa.vNeighbors), a));
        }

        va.factors.clear();
        va.vNeighbors.clear();
        va.active = false;
        --activeVariables_;
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

    template <typename FD>
    size_t FactorGraph<FD>::bestVariableToRemove(const Factors & F) const {
        if (activeVariables_ == 0) return 0;

        // Find first active variable
        size_t retval = 0;
        while (!variableAdjacencies_[retval].active) ++retval;

        // Find the neighbors of this variable, and whether there's a factor
        // with all of them.
        const auto & vNeighbors = getVariables(retval);

        // We want the variable with the minimum size factor, where the factor
        // already exists (so we don't have to allocate anything).
        bool factorExists = false;
        if (vNeighbors.size() > 0) {
            const auto factorIt = findFactorByVariables(variableAdjacencies_[vNeighbors[0]].factors, vNeighbors);
            factorExists = factorIt != std::end(variableAdjacencies_[vNeighbors[0]].factors);
        }

        size_t minCost = F[retval];
        for (auto n : vNeighbors)
            minCost *= F[n];

        for (size_t next = retval + 1; next < variableAdjacencies_.size(); ++next) {
            if (!variableAdjacencies_[next].active)
                continue;

            const auto & vNeighbors = getVariables(next);

            bool newExists = false;
            if (vNeighbors.size() > 0) {
                const auto factorIt = findFactorByVariables(variableAdjacencies_[vNeighbors[0]].factors, vNeighbors);
                newExists = factorIt != std::end(variableAdjacencies_[vNeighbors[0]].factors);
            }

            // If we already have a factor, there's no point in looking at this
            // variable.
            if (!newExists && factorExists) continue;

            // Otherwise compute its cost
            size_t newCost = F[next];
            for (auto n : vNeighbors)
                newCost *= F[n];

            // If we didn't have a factor, or the new cost is less than the old
            // one, we select this variable.
            if ((newExists && !factorExists) || (newCost < minCost)) {
                retval = next;
                minCost = newCost;
            }
        }
        return retval;
    }
}

#endif
