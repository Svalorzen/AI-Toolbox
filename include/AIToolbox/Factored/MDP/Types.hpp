#ifndef AI_TOOLBOX_FACTORED_MDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_TYPES_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>

#include <vector>
#include <utility>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This struct represents a factored ValueFunction.
     *
     * A ValueFunction is simply a function that maps states to values. Here,
     * we use a FactoredVector to represent all values. In addition, we include
     * the weights that can be used to modify the ValueFunction without
     * touching the bases; this is done for example in factored ValueIteration,
     * which updates the weights at each update to better approximate V*.
     */
    struct ValueFunction {
        FactoredVector values;
        Vector weights;
    };

    /**
     * @brief This represents a factored QFunction.
     */
    using QFunction = FactoredMatrix2D;

    /**
     * @brief This struct represents a single state/value tuple.
     *
     * This struct can be used to represent factored Value Functions (possibly
     * inside a FactorGraph) or a set of basis functions.
     */
    struct ValueFunctionRule {
        PartialState state;
        double value;

        ValueFunctionRule(PartialState s, double v) :
                state(std::move(s)), value(v) {}
    };

    /**
     * @brief This struct represents a single state/action/value tuple.
     *
     * This struct can be used in place of a full-blown QFunction matrix
     * when the QFunction matrix would be sparse. Instead, only interesting
     * state/action/value tuples are stored and acted upon.
     */
    struct QFunctionRule {
        PartialState state;
        PartialAction action;
        double value;

        QFunctionRule(PartialState s, PartialAction a, double v) :
                state(std::move(s)), action(std::move(a)), value(v) {}
    };

    /**
     * @brief This struct represents a single state/action/values tuple.
     *
     * This struct can be used in place of a full-blown QFunction matrix for
     * multi-objective MDPs. Thus each state-action pair is linked with a
     * vector of rewards, one for each possible MDP objective.
     */
    struct MOQFunctionRule {
        PartialState state;
        PartialAction action;
        Rewards values;

        MOQFunctionRule(PartialState s, PartialAction a, Rewards vs) :
                state(std::move(s)), action(std::move(a)), values(std::move(vs)) {}
    };
}

#endif
