#include <AIToolbox/FactoredMDP/Types.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        using FactoredState = std::vector<size_t>;
        using PartialState = std::vector<std::pair<size_t, size_t>>;
    }
}
