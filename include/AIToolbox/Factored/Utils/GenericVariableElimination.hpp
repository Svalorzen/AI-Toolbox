#ifndef AI_TOOLBOX_FACTORED_GENERIC_VARIABLE_ELIMINATION_HEADER_FILE
#define AI_TOOLBOX_FACTORED_GENERIC_VARIABLE_ELIMINATION_HEADER_FILE

#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This class represents the Variable Elimination algorithm.
     *
     * This class applies Variable Elimination to an input FactorGraph<Factor>.
     *
     * Since the cross-sum steps in the algorithm differ from the type of node
     * in the graph, we require as input a separate structure which may contain
     * certain methods depending on what the use-case requires, and which holds
     * any needed temporaries to store for the duration of the algorithm.
     *
     * In particular, this structure (the `global` parameter), *MUST* provide:
     *
     * - A member `Factor newFactor` which stores the results of the cross-sum
     *   of each removed variable. At each iteration over the values of that
     *   variable's neighbors, we move from it, so be sure to re-initialize it
     *   if needed.
     * - A member `void crossSum(const Factor &)` function, which should
     *   perform the cross-sum of the input into the `newFactor` member
     *   variable.
     * - A member `void makeResult(FinalFactors &&)` method, which should
     *   process the final factors of the VE process in order to create your
     *   result.
     *
     * Since VE usually requires custom computations, you can *OPTIONALLY*
     * define the following methods:
     *
     * - A member `void beginRemoval(const Graph &, const Graph::FactorItList
     *   &, const Graph::VariableList &, size_t)` method, which is called
     *   at the beginning of the removal of each variable.
     * - A member `void initNewFactor()` method, which is called when the
     *   `newFactor` variable needs to be initialized.
     * - A member `void beginCrossSum(size_t)` method, which is called at the
     *   beginning of each set of cross-sum operations with the current value
     *   of the variable being eliminated.
     * - A member `void endCrossSum()` method, which is called at the end of
     *   each set of cross-sum operations.
     * - A member `bool isValidNewFactor()` method, which returns whether the
     *   `newFactor` variable can be used after all cross-sum operations.
     * - A member `void mergeRules(Rules &&, Rules &&)` method, which can be
     *   used to specify a custom step during the merge of the rules created by
     *   eliminating a variable with the previous ones.
     *
     * All these functions can optionally be `const`; nothing changes. The
     * class will fail to compile if you provide a method with the required
     * name but with the wrong signature, as we would just skip it silently
     * otherwise.
     *
     * @tparam Factor The Factor type to use.
     */
    template <typename Factor>
    class GenericVariableElimination {
        public:
            using Rule = std::pair<PartialValues, Factor>;
            using Rules = std::vector<Rule>;
            using Graph = FactorGraph<Rules>;
            using FinalFactors = std::vector<Factor>;

            /**
             * @brief This operator performs the Variable Elimination operation on the inputs.
             *
             * @param F The space of all factors to eliminate.
             * @param graph The already populated graph to perform VE onto.
             * @param global The global callback structure.
             */
            template <typename Global>
            void operator()(const Factors & F, Graph & graph, Global & global);

        private:
            /**
             * @brief An helper struct to validate the interface of the global callback structure.
             *
             * @tparam M The type of the global callback structure to validate.
             */
            template <typename M>
            struct global_interface;

            /**
             * @brief This function removes the input factor from the graph.
             *
             * @param F The space of all factors to eliminate.
             * @param graph The already populated graph to perform VE onto.
             * @param f The factor to eliminate.
             * @param finalFactors The storage of all the eliminated factors with no remaining neighbors.
             * @param global The global callback structure.
             */
            template <typename Global>
            void removeFactor(const Factors & F, Graph & graph, const size_t f, FinalFactors & finalFactors, Global & global);
    };

    template <typename Factor>
    template <typename M>
    struct GenericVariableElimination<Factor>::global_interface {
        private:
            #define STR2(X) #X
            #define STR(X) STR2(X)
            #define ARG(...) __VA_ARGS__

            // For each function we want to check, we are going to try each
            // overload in succession (char->int->long->...).
            //
            // The first two simply accept the function with the approved
            // signature, whether it is const or not. The third checks whether
            // the member just exists, and reports that it probably has the
            // wrong signature (since we didn't match before).
            //
            // The last just fails to find the match.
            #define MEMBER_CHECK(name, retval, input)                                       \
                                                                                            \
            template <typename Z> static constexpr auto name##Check(char) -> decltype(      \
                    static_cast<retval (Z::*)(input)> (&Z::name),                           \
                    bool()                                                                  \
                    ) { return true; }                                                      \
            template <typename Z> static constexpr auto name##Check(int) -> decltype(       \
                    static_cast<retval (Z::*)(input) const> (&Z::name),                     \
                    bool()                                                                  \
                    ) { return true; }                                                      \
            template <typename Z> static constexpr auto name##Check(long) -> decltype(      \
                    &Z::name,                                                               \
                    bool())                                                                 \
                    {                                                                       \
                        static_assert(!std::is_same_v<M, M>, "You provide a member '" STR(name) "' but with the wrong signature."); \
                        return false;                                                       \
                    }                                                                       \
            template <typename Z> static constexpr auto name##Check(...) -> bool { return false; }

            MEMBER_CHECK(beginRemoval, void, ARG(const Graph &, const typename Graph::FactorItList &, const typename Graph::Variables &, size_t))
            MEMBER_CHECK(initNewFactor, void, void)
            MEMBER_CHECK(beginCrossSum, void, size_t)
            MEMBER_CHECK(crossSum, void, const Factor &)
            MEMBER_CHECK(endCrossSum, void, void)
            MEMBER_CHECK(isValidNewFactor, bool, void)
            MEMBER_CHECK(mergeRules, Rules, ARG(Rules &&, Rules &&))
            MEMBER_CHECK(makeResult, void, FinalFactors &&)

            #undef MEMBER_CHECK
            #undef ARG
            #undef STR
            #undef STR2

        public:
            // All results are stored here for use later. All optional members
            // that do not exist, we simply do not call.
            enum {
                beginRemoval     = beginRemovalCheck<M>     ('\0'),
                initNewFactor    = initNewFactorCheck<M>    ('\0'),
                beginCrossSum    = beginCrossSumCheck<M>    ('\0'),
                crossSum         = crossSumCheck<M>         ('\0'),
                endCrossSum      = endCrossSumCheck<M>      ('\0'),
                isValidNewFactor = isValidNewFactorCheck<M> ('\0'),
                mergeRules       = mergeRulesCheck<M>       ('\0'),
                makeResult       = makeResultCheck<M>       ('\0'),
            };
    };

    template <typename Factor>
    template <typename Global>
    void GenericVariableElimination<Factor>::operator()(const Factors & F, Graph & graph, Global & global) {
        static_assert(global_interface<Global>::crossSum, "You must provide a crossSum method!");
        static_assert(global_interface<Global>::makeResult, "You must provide a makeResult method!");
        static_assert(std::is_same_v<Factor, decltype(global.newFactor)>, "You must provide a public 'Factor newFactor;' member!");

        FinalFactors finalFactors;

        // This can possibly be improved with some heuristic ordering
        while (graph.variableSize())
            removeFactor(F, graph, graph.variableSize() - 1, finalFactors, global);

        global.makeResult(std::move(finalFactors));
    }

    template <typename Factor>
    template <typename Global>
    void GenericVariableElimination<Factor>::removeFactor(const Factors & F, Graph & graph, const size_t f, FinalFactors & finalFactors, Global & global) {
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

            if constexpr(global_interface<Global>::initNewFactor)
                global.initNewFactor();

            for (size_t sAction = 0; sAction < F[f]; ++sAction) {
                if constexpr(global_interface<Global>::beginCrossSum)
                    global.beginCrossSum(sAction);

                jointAction.second[id] = sAction;
                for (const auto factor : factors)
                    for (const auto rule : factor->getData())
                        if (match(factor->getVariables(), rule.first, jointAction.first, jointAction.second))
                            global.crossSum(rule.second);

                if constexpr(global_interface<Global>::endCrossSum)
                    global.endCrossSum();
            }

            bool isValidNewFactor = true;
            if constexpr(global_interface<Global>::isValidNewFactor)
                isValidNewFactor = global.isValidNewFactor();

            if (isValidNewFactor) {
                if (!isFinalFactor) {
                    newRules.emplace_back(jointAction.second, std::move(global.newFactor));
                    // Remove new agent ID
                    newRules.back().first.erase(newRules.back().first.begin() + id);
                }
                else
                    finalFactors.push_back(std::move(global.newFactor));
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

            auto & newFactorNode = graph.getFactor(variables)->getData();

            if constexpr(global_interface<Global>::mergeRules) {
                newFactorNode = global.mergeRules(std::move(newFactorNode), std::move(newRules));
            } else {
                newFactorNode.insert(
                    std::end(newFactorNode),
                    std::make_move_iterator(std::begin(newRules)),
                    std::make_move_iterator(std::end(newRules))
                );
            }
        }
    }
}

#endif
