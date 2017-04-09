#include <AIToolbox/MDP/Policies/QPolicyInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        QPolicyInterface::QPolicyInterface(const QFunction & q) :
                PolicyInterface(q.rows(), q.cols()), q_(q) {}

        const QFunction & QPolicyInterface::getQFunction() const {
            return q_;
        }
    }
}
