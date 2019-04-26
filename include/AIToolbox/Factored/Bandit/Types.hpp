#ifndef AI_TOOLBOX_FACTORED_BANDIT_TYPES_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_TYPES_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>

#include <vector>
#include <utility>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This struct represents a single action/value pair.
     *
     * This struct can be used in place of a full-blown QFunction matrix
     * when the QFunction matrix would be sparse. Instead, only intresting
     * action/value pairs are stored and acted upon.
     */
    struct QFunctionRule {
        PartialAction action;
        double value;

        QFunctionRule(PartialAction a, double v) :
                action(std::move(a)), value(v) {}
    };

    /**
     * @brief This struct represents a single action/values pair.
     *
     * This struct can be used in place of a full-blown QFunction matrix for
     * multi-objective bandits. Thus each action is linked with a vector of
     * rewards, one for each possible objective.
     */
    struct MOQFunctionRule {
        PartialAction action;
        Rewards values;

        MOQFunctionRule(PartialAction a, Rewards vs) :
                action(std::move(a)), values(std::move(vs)) {}
    };

    /**
     * @brief This represents a factored QFunction.
     */
    using QFunction = FactoredVector;
}

#endif
