#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/MDP/Policies/Policy.hpp>

namespace AIToolbox {
    namespace MDP {
        QFunction makeQFunction(size_t S, size_t A) {
            auto retval = QFunction(S, A);
            retval.fill(0.0);
            return retval;
        }

        ValueFunction makeValueFunction(size_t S) {
            auto values = Values(S);
            values.fill(0.0);
            return std::make_tuple(values, Actions(S, 0));
        }
    }
}
