#include <AIToolbox/MDP/Policies/QPolicyInterface.hpp>

namespace AIToolbox::MDP {
    QPolicyInterface::QPolicyInterface(const QFunction & q) : q_(q) {}

    const QFunction & QPolicyInterface::getQFunction() const {
        return q_;
    }
}
