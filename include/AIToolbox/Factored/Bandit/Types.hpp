#ifndef AI_TOOLBOX_FACTORED_BANDIT_TYPES_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_TYPES_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Factored/Types.hpp>

#include <vector>
#include <utility>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This struct represents a single action/value pair.
     *
     * This struct can be used in place of a full-blown QFunction table
     * when the QFunction matrix would be sparse. Instead, only intresting
     * action/value pairs are stored and acted upon.
     */
    struct QFunctionRule {
        PartialAction a_;
        double value_;

        QFunctionRule(PartialAction a, double v) :
                a_(std::move(a)), value_(v) {}
    };
}

#endif
