#ifndef AI_TOOLBOX_FACTORED_MDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_TYPES_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Factored/Types.hpp>

#include <vector>
#include <utility>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This struct represents a single state/value tuple.
     *
     * This struct can be used to represent factored Value Functions (possibly
     * inside a FactorGraph) or a set of basis functions.
     */
    struct ValueFunctionRule {
        PartialState s_;
        double value_;

        ValueFunctionRule(PartialState s, double v) :
                s_(std::move(s)), value_(v) {}
    };

    /**
     * @brief This struct represents a single state/action/value tuple.
     *
     * This struct can be used in place of a full-blown QFunction table
     * when the QFunction matrix would be sparse. Instead, only intresting
     * state/action/value tuples are stored and acted upon.
     */
    struct QFunctionRule {
        PartialState s_;
        PartialAction a_;
        double value_;

        QFunctionRule(PartialState s, PartialAction a, double v) :
                s_(std::move(s)), a_(std::move(a)), value_(v) {}
    };

    /**
     * @brief This struct represents a single state/action/values tuple.
     *
     * This struct can be used in place of a full-blown QFunction table for
     * multi-objective MDPs. Thus each state-action pair is linked with a
     * vector of rewards, one for each possible MDP objective.
     */
    struct MOQFunctionRule {
        PartialState s_;
        PartialAction a_;
        Rewards values_;

        MOQFunctionRule(PartialState s, PartialAction a, Rewards vs) :
                s_(std::move(s)), a_(std::move(a)), values_(std::move(vs)) {}
    };
}

#endif
