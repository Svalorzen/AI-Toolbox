#ifndef AI_TOOLBOX_FACTORED_BANDIT_GRAPH_UTILS_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_GRAPH_UTILS_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Bandit/TypeTraits.hpp>

#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/LocalSearch.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/MaxPlus.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/ReusingIterativeLocalSearch.hpp>

namespace AIToolbox::Factored::Bandit {
    // This file was designed to let users easily swap between different
    // Maximizers when dealing with factored Bandit functions. With Maximizers
    // we mean the algorithms designed to maximize over a factored function via
    // factored graphs (VariableElimination, MaxPlus, LocalSearch, etc). A
    // common example of this is with the QGreedyPolicy.
    //
    // The functors in this file implement the swapping mechanism, by providing
    // a way to initialize the appropriate FactorGraph from a set of data
    // structures in a somewhat generic manner.
    //
    // The system has been designed to be easily extensible, in the event that
    // either this library or a user introduces a new type that needs to be
    // maximized over.
    //
    // If that is the case, one only needs to specialize the
    // Make/UpdateGraphImpl classes for the needed Maximizers and data types,
    // and no other code will need to change. The specializations will need to
    // provide an operator() with the appropriate signature.
    //
    // Note that this system, being templatized, only allows swapping Maximizer
    // at compile time. At the same time, it should at least allow to not
    // having to rewrite lots of code every single time a change is desired.

    /**
     * @brief This class clumps all implementations that create graphs for data for certain Maximizers.
     */
    template <typename Maximizer, typename Data>
    struct MakeGraphImpl;

    /**
     * @brief This class clumps all implementations that update graphs with data for certain Maximizers.
     */
    template <typename Maximizer, typename Data>
    struct UpdateGraphImpl;

    /**
     * @brief This class is the public interface for initializing the graph in generic code that uses the maximizers.
     *
     * This functor creates a new graph that has the correct factor structure,
     * given the input data, to be accepted by the specified Maximizer type.
     *
     * Note that the graph, after being constructed, will not contain any data.
     * Only the structure is initialized. To update the data contained by the
     * graph, one must use the UpdateGraph functor.
     *
     * A graph will in general only need to be constructed once, but can be
     * updated infinitely. With some Maximizers this can save a lot of work.
     *
     * @tparam Maximizer The type of the maximizer to construct a graph for.
     */
    template <typename Maximizer>
    struct MakeGraph {
        template <typename Data, typename... Args>
        auto operator()(const Data & d, Args && ...args) {
            return MakeGraphImpl<Maximizer, Data>()(d, std::forward<Args>(args)...);
        }
    };

    /**
     * @brief This class is the public interface for updating the input graph with the input data in generic code that uses the maximizers.
     *
     * This functor takes as input a graph that has been created via the
     * MakeGraph functor. It then copies the input Data to the appropriate
     * factors of the graph.
     *
     * This functor can be used multiple times on the same graph.
     * A graph will in general only need to be constructed once, but can be
     * updated infinitely. With some Maximizers this can save a lot of work.
     *
     * @tparam Maximizer The type of the maximizer to construct a graph for.
     */
    template <typename Maximizer>
    struct UpdateGraph {
        template <typename Data, typename... Args>
        void operator()(typename Maximizer::Graph & graph, const Data & d, Args && ...args) {
            UpdateGraphImpl<Maximizer, Data>()(graph, d, std::forward<Args>(args)...);
        }
    };

    // ############################
    // ### VARIABLE ELIMINATION ###
    // ############################

    // Since VE deletes its graph at each update, its associated MakeGraph
    // simply doesn't do any work, and the UpdateGraphs just reconstruct it
    // from scratch every time.

    template <typename Data>
    struct MakeGraphImpl<VariableElimination, Data> {
        typename VariableElimination::Graph operator()(const Data &, const Action &) {
            return VariableElimination::Graph(0);
        }
    };

    template <QFRuleRange Iterable>
    struct UpdateGraphImpl<VariableElimination, Iterable> {
        using VE = VariableElimination;

        void operator()(VE::Graph & graph, const Iterable & inputRules, const Action & A) {
            graph.reset(A.size());

            for (const auto & rule : inputRules) {
                auto & factorNode = graph.getFactor(rule.action.first)->getData();
                const auto id = toIndexPartial(A, rule.action);

                const auto it = std::lower_bound(
                    std::begin(factorNode), std::end(factorNode), id,
                    [](const auto & rule, size_t rhs) {return rule.first < rhs;}
                );

                if (it != std::end(factorNode) && it->first == id)
                    it->second.first += rule.value;
                else
                    factorNode.emplace(it, id, VE::Factor{rule.value, {}});
            }
        }
    };

    template <>
    struct UpdateGraphImpl<VariableElimination, QFunction> {
        using VE = VariableElimination;

        void operator()(VE::Graph & graph, const QFunction & qf, const Action & A) {
            graph.reset(A.size());

            for (const auto & basis : qf.bases) {
                const auto Ai = static_cast<size_t>(basis.values.size());

                auto & factorNode = graph.getFactor(basis.tag)->getData();

                if (factorNode.empty()) {
                    factorNode.reserve(Ai);
                    for (size_t ai = 0; ai < Ai; ++ai)
                        factorNode.emplace_back(ai, VE::Factor{0.0, {}});
                }

                for (size_t ai = 0; ai < Ai; ++ai)
                    factorNode[ai].second.first += basis.values(ai);
            }
        }
    };

    // ###################################
    // ## LOCAL SEARCH / MAXPLUS / RILS ##
    // ###################################

    template <QFRuleRange Iterable>
    struct MakeGraphImpl<LocalSearch, Iterable> {
        typename LocalSearch::Graph operator()(const Iterable & inputRules, const Action & A) {
            LocalSearch::Graph graph(A.size());

            for (const auto & rule : inputRules) {
                auto & factorNode = graph.getFactor(rule.action.first)->getData();

                if (!factorNode.size())
                    factorNode.resize(factorSpacePartial(rule.action.first, A));
            }
            return graph;
        }
    };

    template <>
    struct MakeGraphImpl<LocalSearch, QFunction> {
        typename LocalSearch::Graph operator()(const QFunction & qf, const Action & A) {
            LocalSearch::Graph graph(A.size());

            for (const auto & basis : qf.bases) {
                auto & factorNode = graph.getFactor(basis.tag)->getData();

                if (!factorNode.size())
                    factorNode.resize(basis.values.size());
            }

            return graph;
        }
    };

    template <QFRuleRange Iterable>
    struct UpdateGraphImpl<LocalSearch, Iterable> {
        void operator()(LocalSearch::Graph & graph, const Iterable & inputRules, const Action & A) {
            for (auto & f : graph)
                f.getData().setZero();

            for (const auto & rule : inputRules) {
                auto & factorNode = graph.getFactor(rule.action.first)->getData();

                const auto id = toIndexPartial(A, rule.action);
                factorNode[id] += rule.value;
            }
        }
    };

    template <>
    struct UpdateGraphImpl<LocalSearch, QFunction> {
        void operator()(LocalSearch::Graph & graph, const Factored::Bandit::QFunction & qf, const Action &) {
            for (auto & f : graph)
                f.getData().setZero();

            for (const auto & basis : qf.bases)
                graph.getFactor(basis.tag)->getData() += basis.values;

        }
    };

    // MaxPlus and RILS both use the same graph type, so we don't need to implement anything more.
    template <> struct MakeGraph<MaxPlus> : MakeGraph<LocalSearch> {};
    template <> struct UpdateGraph<MaxPlus> : UpdateGraph<LocalSearch> {};

    template <> struct MakeGraph<ReusingIterativeLocalSearch> : MakeGraph<LocalSearch> {};
    template <> struct UpdateGraph<ReusingIterativeLocalSearch> : UpdateGraph<LocalSearch> {};
}

#endif
