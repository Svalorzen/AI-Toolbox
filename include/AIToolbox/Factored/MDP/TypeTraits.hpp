#ifndef AI_TOOLBOX_FACTORED_MDP_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_TYPE_TRAITS_HEADER_FILE

#include <ranges>

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Bandit/TypeTraits.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This concept models the interface for a QFunctionRule.
     *
     * \sa AIToolbox::Factored::Bandit::IsQFunctionRule
     */
    template <typename QR>
    concept IsQFunctionRule = Bandit::IsQFunctionRule<QR> && requires (const QR qr) {
        { qr.state } -> std::convertible_to<PartialState>;
    };

    /**
     * @brief This concept represents a range of QFunctionRules.
     */
    template <typename T>
    concept QFRuleRange = std::ranges::range<T> && IsQFunctionRule<std::ranges::range_value_t<T>>;
}

#endif
