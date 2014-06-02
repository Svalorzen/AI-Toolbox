#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/MDP/Policies/Policy.hpp>

namespace AIToolbox {
    namespace MDP {
        QFunction makeQFunction(size_t S, size_t A) {
            return QFunction(boost::extents[S][A]);
        }

        ValueFunction makeValueFunction(size_t S) {
            return std::make_tuple(Values(S, 0.0), Actions(S, 0));
        }
    }
}
