#include <AIToolbox/MDP/QPolicyInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        QPolicyInterface::QPolicyInterface(const QFunction & q) : PolicyInterface(q.shape()[0], q.shape()[1]), q_(q) {}
    }
}
