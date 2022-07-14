#ifndef AI_TOOLBOX_FACTORED_MDP_GRAPH_UTILS_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_GRAPH_UTILS_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/MDP/TypeTraits.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/GraphUtils.hpp>

namespace AIToolbox::Factored::MDP {
    // The "main" code to make graph is implemented in the Bandit namespace.
    // This file contains an equivalent hierarchy of classes, but within the
    // MDP namespace.
    //
    // The main reason why we need to reimplement MakeGraphImpl,
    // UpdateGraphImpl, MakeGraph and UpdateGraph in this namespace is that the
    // functors for *updating* factored MDPs have necessarily different
    // arguments (in particular, they need the size of the state space and a
    // specific state).
    //
    // This means that any code that wants to use the Make/UpdateGraph
    // mechanism to write generic code for MDPs will necessarily pass arguments
    // that are not compatible with the classes written in the Bandit
    // namespace, so there is no reason to try to keep that code easily
    // reachable from Factored::MDP.
    //
    // For MakeGraph we could have just kept extending the Bandit class (as we
    // only generally need data + size of action space), but then it would be
    // wierd to only duplicate UpdateGraph stuff -- and also because any direct
    // extension would have needed to still be written in the Bandit namespace
    // since specializations are only allowed in the original namespace. Thus,
    // we duplicate MakeGraph[Impl] as well for consistency.
    //
    // At the same time, we try to reuse everything we can, be it with
    // inheritance or directly calling the original function with the required
    // subset of input parameters.

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

    template <typename Data>
    struct MakeGraphImpl<Bandit::VariableElimination, Data> : public Bandit::MakeGraphImpl<Bandit::VariableElimination, Data> {};

    template <MDP::QFRuleRange Iterable>
    struct UpdateGraphImpl<Bandit::VariableElimination, Iterable> {
        using VE = Bandit::VariableElimination;

        void operator()(VE::Graph & graph, const Iterable & inputRules, const State &, const Action & A, const State &) {
            Bandit::UpdateGraph<VE>()(graph, inputRules, A);
        }
    };

    template <>
    struct UpdateGraphImpl<Bandit::VariableElimination, MDP::QFunction> {
        using VE = Bandit::VariableElimination;

        void operator()(VE::Graph & graph, const MDP::QFunction & qf, const State & S, const Action & A, const State & s) {
            graph.reset(A.size());

            for (const auto & basis : qf.bases) {
                const auto Ai = static_cast<size_t>(basis.values.cols());

                auto & factorNode = graph.getFactor(basis.actionTag)->getData();

                if (factorNode.empty()) {
                    factorNode.reserve(Ai);
                    for (size_t ai = 0; ai < Ai; ++ai)
                        factorNode.emplace_back(ai, VE::Factor{0.0, {}});
                }

                const size_t si = toIndexPartial(basis.tag, S, s);
                for (size_t ai = 0; ai < Ai; ++ai)
                    factorNode[ai].second.first += basis.values(si, ai);
            }
        }
    };

    // ###################################
    // ## LOCAL SEARCH / MAXPLUS / RILS ##
    // ###################################

    template <MDP::QFRuleRange Iterable>
    struct MakeGraphImpl<Bandit::LocalSearch, Iterable> : public Bandit::MakeGraphImpl<Bandit::LocalSearch, Iterable> {};

    template <>
    struct MakeGraphImpl<Bandit::LocalSearch, MDP::QFunction> {
        Bandit::LocalSearch::Graph operator()(const MDP::QFunction & qf, const Action & A) {
            Bandit::LocalSearch::Graph graph(A.size());

            for (const auto & basis : qf.bases) {
                auto & factorNode = graph.getFactor(basis.actionTag)->getData();

                if (!factorNode.size())
                    factorNode.resize(basis.values.cols());
            }

            return graph;
        }
    };

    template <MDP::QFRuleRange Iterable>
    struct UpdateGraphImpl<Bandit::LocalSearch, Iterable> {
        using LS = Bandit::LocalSearch;

        void operator()(LS::Graph & graph, const Iterable & inputRules, const State &, const Action & A, const State &) {
            Bandit::UpdateGraph<LS>()(graph, inputRules, A);
        }
    };

    template <>
    struct UpdateGraphImpl<Bandit::LocalSearch, MDP::QFunction> {
        void operator()(Bandit::LocalSearch::Graph & graph, const MDP::QFunction & qf, const State & S, const Action &, const State & s) {
            for (auto & f : graph)
                f.getData().setZero();

            for (const auto & basis : qf.bases) {
                const size_t si = toIndexPartial(basis.tag, S, s);
                graph.getFactor(basis.actionTag)->getData() += basis.values.row(si);
            }
        }
    };

    // MaxPlus and RILS both use the same graph type, so we don't need to implement anything more.
    template <> struct MakeGraph<Bandit::MaxPlus> : MakeGraph<Bandit::LocalSearch> {};
    template <> struct UpdateGraph<Bandit::MaxPlus> : UpdateGraph<Bandit::LocalSearch> {};

    template <> struct MakeGraph<Bandit::ReusingIterativeLocalSearch> : MakeGraph<Bandit::LocalSearch> {};
    template <> struct UpdateGraph<Bandit::ReusingIterativeLocalSearch> : UpdateGraph<Bandit::LocalSearch> {};
}

#endif
