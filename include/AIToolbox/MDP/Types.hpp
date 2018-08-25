#ifndef AI_TOOLBOX_MDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_MDP_TYPES_HEADER_FILE

#include <vector>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/TypeTraits.hpp>

namespace AIToolbox::MDP {
    /**
     * @name MDP Value Types
     *
     * QFunctions and ValueFunctions are specific functions that are
     * defined in terms of policies; as in, in any particular state,
     * they can evaluate the performance that the policy will have.
     * In general however here we do not specifically specify what the
     * policy is, and since we are most probably interested in the best
     * possible policy, we try to store as little information as
     * possible in order to find that out.
     *
     * A QFunction is a function that takes in a state and action, and
     * returns the value for that particular pair. The higher the value
     * is, the better we predict we will perform. Using a QFunction to
     * obtain the perfect policy is straightforward, since at each state
     * we can simply check which action will yeld the best value, and
     * choose that one (assuming that all actions taken from that point
     * are optimal, which we would like to assume since we are trying
     * to find out the best).
     *
     * In theory, a ValueFunction is a function that is a max over
     * actions of the QFunction, as in it takes a state and returns
     * the best value obtainable from that state (following the implied
     * policy). However, that is not very useful in a practical scenario.
     * Thus we want to store not only that value, but also the action
     * that resulted in that particular choice. Instead of storing, as
     * it would make more intuitive sense, this function as a vector of
     * tuples, we are going to store it as a tuple of vectors, to allow
     * for easy manipulations of the underlying values (sums, products
     * and so on).
     *
     * @{
     */

    using Values            = Vector;
    using Actions           = std::vector<size_t>;

    struct ValueFunction {
        Values values;
        Actions actions;

        ValueFunction() {}
        ValueFunction(Values v, Actions a) :
                values(std::move(v)), actions(std::move(a)) {}
    };

    using QFunction = Matrix2D;

    /** @}  */
}

#endif
