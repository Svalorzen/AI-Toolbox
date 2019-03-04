#ifndef AI_TOOLBOX_FACTORED_GENERIC_VARIABLE_ELIMINATION_HEADER_FILE
#define AI_TOOLBOX_FACTORED_GENERIC_VARIABLE_ELIMINATION_HEADER_FILE

#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>

namespace AIToolbox::Factored {
    template <typename Factor>
    class GenericVariableElimination {
        public:
            using Rule = std::pair<PartialValues, Factor>;
            using Rules = std::vector<Rule>;
            using Graph = FactorGraph<Rules>;
            using FinalFactors = std::vector<Factor>;

        public:
            template <typename Global>
            void operator()(const Factors & F, Graph & graph, Global & global);

        private:
            template <typename M>
            struct global_interface;

            template <typename Global>
            void removeFactor(const Factors & F, Graph & graph, const size_t f, FinalFactors & finalFactors, Global global);
    };

    template <typename Factor>
    template <typename M>
    struct GenericVariableElimination<Factor>::global_interface {
        private:
            #define STR2(X) #X
            #define STR(X) STR2(X)
            #define ARG(...) __VA_ARGS__

            #define MEMBER_CHECK(name, retval, input)                                   \
                                                                                        \
            template <typename Z> static constexpr auto name##Check(int) -> decltype(   \
                    static_cast<retval (Z::*)(input)> (&Z::name),                       \
                    bool()                                                              \
                    ) { return true; }                                                  \
            template <typename Z> static constexpr auto name##Check(long) -> decltype(  \
                    &Z::name,                                                           \
                    bool())                                                             \
                    {                                                                   \
                        static_assert(!std::is_same_v<M, M>, "You provide a member '" STR(name) "' but with the wrong signature."); \
                        return false;                                                   \
                    }                                                                   \
            template <typename Z> static constexpr auto name##Check(...) -> bool { return false; }

            MEMBER_CHECK(beginRemoval, void, ARG(const Graph &, const typename Graph::FactorItList &, const typename Graph::VariableList &, const size_t))
            MEMBER_CHECK(initFactor, void, void)
            MEMBER_CHECK(beginCrossSum, void, void)
            MEMBER_CHECK(crossSum, void, const Factor &)
            MEMBER_CHECK(endCrossSum, void, void)
            MEMBER_CHECK(isValidFactor, bool, void)
            MEMBER_CHECK(mergeRules, Rules, ARG(Rules &&, Rules &&))
            MEMBER_CHECK(makeResult, void, FinalFactors &&)

            #undef MEMBER_CHECK
            #undef ARG
            #undef STR
            #undef STR2

        public:
            enum {
                beginRemoval    = beginRemovalCheck<M>(0),
                initFactor      = initFactorCheck<M>(0),
                beginCrossSum   = beginCrossSumCheck<M>(0),
                crossSum        = crossSumCheck<M>(0),
                endCrossSum     = endCrossSumCheck<M>(0),
                isValidFactor   = isValidFactorCheck<M>(0),
                mergeRules      = mergeRulesCheck<M>(0),
                makeResult      = makeResultCheck<M>(0),
            };
    };

    template <typename Factor>
    template <typename Global>
    void GenericVariableElimination<Factor>::operator()(const Factors & F, Graph & graph, Global & global) {
        static_assert(global_interface<Global>::crossSum, "You must provide a crossSum method!");
        static_assert(global_interface<Global>::makeResult, "You must provide a makeResult method!");
        static_assert(std::is_same_v<Factor, decltype(global.factor)>, "You must provide a public 'Factor factor;' member!");

        FinalFactors finalFactors;

        // This can possibly be improved with some heuristic ordering
        while (graph.variableSize())
            removeFactor(F, graph, graph.variableSize() - 1, finalFactors, global);

        global.makeResult(std::move(finalFactors));
    }

    template <typename Factor>
    template <typename Global>
    void GenericVariableElimination<Factor>::removeFactor(const Factors & F, Graph & graph, const size_t f, FinalFactors & finalFactors, Global global) {
        const auto factors = graph.getNeighbors(f);
        auto variables = graph.getNeighbors(factors);

        PartialFactorsEnumerator jointActions(F, variables, f);
        const auto id = jointActions.getFactorToSkipId();

        Rules newRules;

        if constexpr(global_interface<Global>::beginRemoval)
            global.beginRemoval(graph, factors, variables, f);

        // We'll now create new rules that represent the elimination of the
        // input variable for this round.
        const bool isFinalFactor = variables.size() == 1;

        while (jointActions.isValid()) {
            auto & jointAction = *jointActions;

            if constexpr(global_interface<Global>::initFactor)
                global.initFactor();

            for (size_t sAction = 0; sAction < F[f]; ++sAction) {
                if constexpr(global_interface<Global>::beginCrossSum)
                    global.beginCrossSum();

                jointAction.second[id] = sAction;
                for (const auto factor : factors)
                    for (const auto rule : factor->getData())
                        if (match(factor->getVariables(), rule.first, jointAction.first, jointAction.second))
                            global.crossSum(rule.second);

                if constexpr(global_interface<Global>::endCrossSum)
                    global.endCrossSum();
            }

            bool isValidFactor = true;
            if constexpr(global_interface<Global>::isValidFactor)
                isValidFactor = global.isValidFactor();

            if (isValidFactor) {
                if (!isFinalFactor) {
                    newRules.emplace_back(jointAction.second, std::move(global.factor));
                    // Remove new agent ID
                    newRules.back().first.erase(newRules.back().first.begin() + id);
                }
                else
                    finalFactors.push_back(std::move(global.factor));
            }
            jointActions.advance();
        }

        // And finally as usual in variable elimination remove the variable
        // from the graph and insert the newly created variable in.

        for (const auto & it : factors)
            graph.erase(it);
        graph.erase(f);

        if (!isFinalFactor && newRules.size()) {
            variables.erase(std::remove(std::begin(variables), std::end(variables), f), std::end(variables));

            auto newFactor = graph.getFactor(variables);

            if constexpr(global_interface<Global>::mergeRules)
                newFactor->getData() = global.mergeRules(std::move(newFactor->getData()), std::move(newRules));
            else {
                newFactor->getData().insert(
                    std::end(newFactor->getData()),
                    std::make_move_iterator(std::begin(newRules)),
                    std::make_move_iterator(std::end(newRules))
                );
            }
        }
    }
}

#endif
