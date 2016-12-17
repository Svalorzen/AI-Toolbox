#include <AIToolbox/MDP/Policies/QPolicyInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        QPolicyInterface::QPolicyInterface(const QFunction & q) :
                PolicyInterface(q.rows(), q.cols()), q_(q) {}
    }
}
