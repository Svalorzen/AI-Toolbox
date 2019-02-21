#ifndef AI_TOOLBOX_FACTORED_TYPES_HEADER_FILE
#define AI_TOOLBOX_FACTORED_TYPES_HEADER_FILE

#include <AIToolbox/Types.hpp>

#include <vector>
#include <utility>

namespace AIToolbox::Factored {
    /**
     * @name Factored Basic Types
     *
     * Here we define alternative representations for states and actions,
     * where they are factored. A factored state/action can be split into
     * multiple sub-components, which are at least partially independent
     * from each other.
     *
     * This allows for two advantages: the first is that we can represent
     * in a better way worlds where the number of states could be
     * incredibly high, but which could very well be described as composed
     * of a limited number of specific features.
     *
     * The other advantage of factorization is that very often rewards
     * depend only on a specific subset of the state or action. Instead,
     * the a reward can be considered as a sum of multiple reward
     * functions, each acting on a specific subset of state and actions.
     *
     * This potentially allows for solving problems more efficiently, as
     * each new reward function is now defined on exponentially less state
     * and actions, and even if we have to take into account more of them,
     * the savings usually justify the factorization.
     *
     * A very useful property of this factorization is also that we can use
     * this exact same methodology to approach cooperative MDPs with
     * multiple agents. Each agent will then become a factor in the newly
     * defined action space.
     *
     * Here we represent a Factor, which would be some number which can be
     * represented through separate factors, as a vector where each
     * component 'i' can take a number from 0 to N_i.
     *
     * Since we are also interested into subsets of these factors, as
     * specified above, we introduce the concept of PartialFactors. This is
     * a pair formed by two equally sized vectors, where the first contains
     * the indeces of the original Factor which are being taken into
     * consideration, and the second vector contains their values.
     *
     * An additional definition which can be useful in case of multi-objective
     * MDPs is the Rewards one, which contains a vector of rewards, one per
     * objective. Multi-objective MDPs happen when there is no established
     * priority between different reward functions at the time of planning, and
     * so there is no way to reduce the value of a given action to a single
     * number. In such a case, a reward is a vector, and each element in the
     * vector will be weighted in the final action choice. Planning however
     * results more complicated as more possible courses of action have to be
     * considered, as there is no way to discard them in advance (not knowing
     * the weights).
     *
     * @{
     */

    using Factors = std::vector<size_t>;
    using PartialKeys = std::vector<size_t>;
    using PartialValues = std::vector<size_t>;
    using PartialFactors = std::pair<PartialKeys, PartialValues>;

    using State = Factors;
    using PartialState = PartialFactors;
    using Action = Factors;
    using PartialAction = PartialFactors;
    using Rewards = Vector;

    // @}
}

#endif
