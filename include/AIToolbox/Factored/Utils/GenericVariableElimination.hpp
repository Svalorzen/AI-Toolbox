#ifndef AI_TOOLBOX_FACTORED_GENERIC_VARIABLE_ELIMINATION_HEADER_FILE
#define AI_TOOLBOX_FACTORED_GENERIC_VARIABLE_ELIMINATION_HEADER_FILE

#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>

#include <AIToolbox/Impl/Logging.hpp>
#include <AIToolbox/Impl/FunctionMatching.hpp>

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
     * - A member `void makeResult(std::vector<Factor> &&)` method, which should
     *   process the final factors of the VE process in order to create your
     *   result.
     *
     * Since VE usually requires custom computations, you can *OPTIONALLY*
     * define the following methods:
     *
     * - A member `void beginRemoval(const Graph &, const Graph::FactorItList
     *   &, const Graph::Variable &, size_t)` method, which is called
     *   at the beginning of the removal of each variable.
     * - A member `void initNewFactor()` method, which is called when the
     *   `newFactor` variable needs to be initialized.
     * - A member `void beginCrossSum(size_t)` method, which is called at the
     *   beginning of each set of cross-sum operations with the current value
     *   of the variable being eliminated.
     * - A member `void beginFactorCrossSum()` method, which is called at the
     *   beginning of each set of cross-sum operations with a given factor.
     * - A member `void endFactorCrossSum()` method, which is called at the end
     *   of each set of cross-sum operations with a given factor.
     * - A member `void endCrossSum()` method, which is called at the end of
     *   each set of cross-sum operations.
     * - A member `bool isValidNewFactor()` method, which returns whether the
     *   `newFactor` variable can be used after all cross-sum operations.
     * - A member `void mergeFactors(Factor &, Factor &&)` function, which
     *   should merge the rhs into the lhs. If not specified a new Rule is
     *   appended to the Rules rather than merged. If this function is
     *   specified the input graph *must* have sorted Rules!!
     *
     * All these functions can optionally be `const`; nothing changes. In
     * addition, for the 'beginRemoval' and 'beginCrossSum' functions, all
     * parameters are optional: you can define only the ones you want (as long
     * as they are in the same order as specified here), and we will call them
     * correctly.
     *
     * The class will fail to compile if you provide a method with the required
     * name but with the wrong signature, as we would just skip it silently
     * otherwise.
     *
     * @tparam Factor The Factor type to use.
     */
    template <typename Factor>
    class GenericVariableElimination {
        public:
            using Rule = std::pair<size_t, Factor>;
            using Rules = std::vector<Rule>;
            using Graph = FactorGraph<Rules>;
            using FinalFactors = std::vector<Factor>;

            /**
             * @brief This operator performs the Variable Elimination operation on the inputs.
             *
             * @param V The space of all variables to eliminate.
             * @param graph The already populated graph to perform VE onto.
             * @param global The global callback structure.
             */
            template <typename Global>
            void operator()(const Factors & V, Graph & graph, Global & global);

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
             * @param V The space of all variables to eliminate.
             * @param graph The already populated graph to perform VE onto.
             * @param v The variable to eliminate.
             * @param finalFactors The storage of all the eliminated factors with no remaining neighbors.
             * @param global The global callback structure.
             */
            template <typename Global>
            void removeFactor(const Factors & V, Graph & graph, const size_t v, FinalFactors & finalFactors, Global & global);
    };

    template <typename Factor>
    template <typename M>
    struct GenericVariableElimination<Factor>::global_interface {
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
        private:                                                                            \
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
                        static_assert(Impl::is_compatible_f<                                \
                                        decltype(&Z::name),                                 \
                                        retval(input)                                       \
                                      >::value, "You provide a member '" STR(name) "' but with the wrong signature."); \
                        return true;                                                        \
                    }                                                                       \
            template <typename Z> static constexpr auto name##Check(...) -> bool { return false; } \
                                                                                            \
        public:                                                                             \
            enum {                                                                          \
                name = name##Check<M>('\0')                                                 \
            };

            MEMBER_CHECK(beginRemoval, void, ARG(const Graph &, const typename Graph::FactorItList &, const typename Graph::Variables &, size_t))
            MEMBER_CHECK(initNewFactor, void, void)
            MEMBER_CHECK(beginCrossSum, void, size_t)
            MEMBER_CHECK(beginFactorCrossSum, void, void)
            MEMBER_CHECK(crossSum, void, const Factor &)
            MEMBER_CHECK(endFactorCrossSum, void, void)
            MEMBER_CHECK(endCrossSum, void, void)
            MEMBER_CHECK(isValidNewFactor, bool, void)
            MEMBER_CHECK(mergeFactors, void, ARG(Factor &, Factor &&))
            MEMBER_CHECK(makeResult, void, FinalFactors &&)

            #undef MEMBER_CHECK
            #undef ARG
            #undef STR
            #undef STR2
    };

    template <typename Factor>
    template <typename Global>
    void GenericVariableElimination<Factor>::operator()(const Factors & V, Graph & graph, Global & global) {
        static_assert(global_interface<Global>::crossSum, "You must provide a crossSum method!");
        static_assert(global_interface<Global>::makeResult, "You must provide a makeResult method!");
        static_assert(std::is_same_v<Factor, decltype(global.newFactor)>, "You must provide a public 'Factor newFactor;' member!");

        FinalFactors finalFactors;

        // We remove variables one at a time from the graph, storing the last
        // remaining nodes in the finalFactors variable.
        while (graph.variableSize())
            removeFactor(V, graph, graph.bestVariableToRemove(V), finalFactors, global);

        global.makeResult(std::move(finalFactors));
    }

    template <typename Factor>
    template <typename Global>
    void GenericVariableElimination<Factor>::removeFactor(const Factors & V, Graph & graph, const size_t v, FinalFactors & finalFactors, Global & global) {
        AI_LOGGER(AI_SEVERITY_INFO, "Removing variable " << v);

        // We iterate over all possible joint values of the neighbors of 'f';
        // these are all variables which share at least one factor with it.
        const auto & factors = graph.getFactors(v);
        const auto & vNeighbors = graph.getVariables(v);

        if constexpr(global_interface<Global>::beginRemoval)
            Impl::callFunction(global, &Global::beginRemoval, graph, factors, vNeighbors, v);

        // We'll now create new rules that represent the elimination of the
        // input variable for this round.
        const bool isFinalFactor = vNeighbors.size() == 0;

        Rules * oldRulesP;
        size_t oldRulesCurrId = 0;

        PartialFactorsEnumerator jointValues(V, vNeighbors, v, true);
        const auto id = jointValues.getFactorToSkipId();

        if (!isFinalFactor) {
            oldRulesP = &graph.getFactor(vNeighbors)->getData();
            oldRulesP->reserve(jointValues.size());
        }

        AI_LOGGER(
            AI_SEVERITY_DEBUG,
            "Width of this factor: " << vNeighbors.size() + 1 << ". "
            "Joint values to iterate: " << jointValues.size() * V[v]
        );

        size_t jvID = 0;
        while (jointValues.isValid()) {
            auto & jointValue = *jointValues;

            if constexpr(global_interface<Global>::initNewFactor)
                global.initNewFactor();

            // Since we are eliminating 'v', we iterate over its possible
            // values and we reduce over them; this could be a cross-sum
            // operation, a max, or anything else.
            for (size_t vValue = 0; vValue < V[v]; ++vValue) {
                if constexpr(global_interface<Global>::beginCrossSum)
                    Impl::callFunction(global, &Global::beginCrossSum, vValue);

                jointValue.second[id] = vValue;
                for (const auto factor : factors) {
                    if constexpr(global_interface<Global>::beginFactorCrossSum)
                        global.beginFactorCrossSum();

                    // We reduce over each Factor that is applicable to this
                    // particular joint value set.
                    const size_t jvPartialIndex = toIndexPartial(factor->getVariables(), V, jointValue);
                    if constexpr(global_interface<Global>::mergeFactors) {
                        const auto & data = factor->getData();
                        const auto ruleIt = std::lower_bound(
                            std::begin(data),
                            std::end(data),
                            jvPartialIndex,
                            [](const Rule & lhs, const size_t rhs) {
                                return lhs.first < rhs;
                            }
                        );
                        if (ruleIt != std::end(data) && ruleIt->first == jvPartialIndex)
                            global.crossSum(ruleIt->second);
                    } else {
                        for (const auto rule : factor->getData())
                            if (jvPartialIndex == rule.first)
                                global.crossSum(rule.second);
                    }

                    if constexpr(global_interface<Global>::endFactorCrossSum)
                        global.endFactorCrossSum();
                }

                if constexpr(global_interface<Global>::endCrossSum)
                    global.endCrossSum();
            }

            bool isValidNewFactor = true;
            if constexpr(global_interface<Global>::isValidNewFactor)
                isValidNewFactor = global.isValidNewFactor();

            // If the new Factor is good, we save it together with the joint
            // value that has produced it (minus the one of the variable to
            // remove). If it has no neighbors, we add it to the finalFactors
            // instead.
            if (isValidNewFactor) {
                if (!isFinalFactor) {
                    auto & oldRules = *oldRulesP;

                    // If we care enough to merge, we store all rules in
                    // lexicographical order of value; if the old rules already
                    // contained this same value and we are provided with a
                    // merge function, we can merge the two, otherwise we
                    // insert it as-is in the correct spot.
                    if constexpr(global_interface<Global>::mergeFactors) {
                        while (oldRulesCurrId < oldRules.size() && oldRules[oldRulesCurrId].first < jvID)
                            ++oldRulesCurrId;

                        if (oldRulesCurrId < oldRules.size() && oldRules[oldRulesCurrId].first == jvID) {
                            global.mergeFactors(oldRules[oldRulesCurrId].second, std::move(global.newFactor));
                        } else {
                            oldRules.emplace(std::begin(oldRules) + oldRulesCurrId, jvID, std::move(global.newFactor));
                        }
                    } else {
                        // Otherwise we simply append, as it should be faster.
                        // Remember, a factor may be appended on multiple
                        // times, but it's only iterated over once before being
                        // removed.
                        oldRules.emplace_back(jvID, std::move(global.newFactor));
                    }
                    ++oldRulesCurrId;
                }
                else
                    finalFactors.push_back(std::move(global.newFactor));
            }
            ++jvID;
            jointValues.advance();
        }

        // And finally we remove the variable from the graph.
        graph.erase(v);
    }
}

#endif
