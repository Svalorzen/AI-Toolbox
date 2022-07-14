#ifndef AI_TOOLBOX_FACTORED_BANDIT_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_TYPE_TRAITS_HEADER_FILE

#include <ranges>

#include <AIToolbox/Factored/Bandit/Types.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This concept models the interface for a QFunctionRule.
     *
     * This is needed so we can consider MDP QFunctionRules as Bandit QFunctionRules, without having to use inheritance to connect them.
     */
    template <typename QR>
    concept IsQFunctionRule = requires (const QR qr) {
        { qr.action } -> std::convertible_to<PartialAction>;
        { qr.value } -> std::convertible_to<double>;
    };

    /**
     * @brief This concept represents a range of QFunctionRules.
     */
    template <typename T>
    concept QFRuleRange = std::ranges::range<T> && IsQFunctionRule<std::ranges::range_value_t<T>>;
}

#endif
