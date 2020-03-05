#ifndef AI_TOOLBOX_POMDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_POMDP_TYPES_HEADER_FILE

#include <utility>
#include <vector>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This represents a belief, which is a probability distribution over states.
     */
    using Belief = ProbabilityVector;

    /**
     * @name POMDP Value Types
     *
     * POMDP ValueFunctions are complicated. In fact, they are trees. At each
     * belief, you have a particular value function, which also depends on any
     * future observation you might encounter, and must change "value" accordingly.
     * For example, a high valued belief might in the end turn out bad due to
     * repeated "bad" observations. At the same time, for each particular block
     * of values, we want to save the single action that will result in the
     * "actuation" of that particular value.
     *
     * We avoid storing a POMDP ValueFunction as a true tree, mostly due to the
     * fact that most operations like search and update are done on a timestep
     * basis, as in, specific tree depth. Thus the layout is arranged as follows:
     *
     * A VEntry contains:
     *
     * - The MDP::Values for its specific Belief range. This is also called an
     *   alphavector in the literature. At any belief it can be used to
     *   compute, via dot product, the true value of that belief.
     * - An action index, for the action that results in the actuation of those
     *   particular values.
     * - A vector containing, for each possible observation, the index of the
     *   VEntry to look into for the next timestep/VList. Thus, there are going
     *   to be |O| entries in this vector (sometimes it's empty, when it
     *   doesn't matter). Some observations are however impossible from certain
     *   beliefs. In theory, those vector entries should never be accessed, so
     *   they will just keep the value of zero to keep things simple.
     *
     * A VList is a slice of the final tree with respect to depth, as in all
     * ValueFunctions for a certain timestep t. Note that a VList can have an
     * arbitrary number of VEntries inside - with an upper bound. Each VList
     * can have at most A * size(VList_{t-1})^O.
     *
     * A ValueFunction is the final tree keeping all VLists together. A
     * ValueFunction has always at least one element.
     *
     * The first element of a ValueFunction is technically useless, as it is a
     * VList with just one VEntry that tells to perform action zero. It's the
     * default from which all Dynamic Programming algorithm start. The values
     * of the default entry are usually zeros, although some algorithms
     * initialize them differently. The first entry otherwise is never used,
     * not even for sampling for a policy, and it's simply an artifact that
     * takes little space to keep, and it's expected in all the code.
     *
     * The UpperBoundValueFunction is a pair of two, equally sized, lists. The
     * first list contains a set of Beliefs where an upper bound is known, and
     * the second contains those upper bounds. It's possible to infer upper
     * bounds outide of the listed Beliefs by interpolation (either sawtooth or
     * LP). This list usually does not contain the corner Beliefs.
     *
     * QFunctions may be defined later, however since POMDP ValueFunctions are already
     * pretty costly in terms of space, in general there's little sense in storing them.
     *
     * @{
     */

    using VObs          = std::vector<size_t>;
    struct VEntry {
        MDP::Values values;
        size_t action;
        VObs observations;

        VEntry() {}
        VEntry(MDP::Values v, size_t a, VObs o) :
                values(std::move(v)), action(a), observations(std::move(o)) {}
        VEntry(size_t S, size_t a, size_t O) :
                values(S), action(a), observations(O) { values.setZero(); }
    };
    using VList         = std::vector<VEntry>;
    using ValueFunction = std::vector<VList>;

    using UpperBoundValueFunction = std::pair<std::vector<Belief>, std::vector<double>>;

    /** @}  */
}

#endif
