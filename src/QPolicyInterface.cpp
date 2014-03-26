#include <AIToolbox/MDP/Policies/QPolicyInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        QPolicyInterface::QPolicyInterface(const QFunction & q) : PolicyInterface<size_t>(q.shape()[0], q.shape()[1]), q_(q) {}
    }
}
