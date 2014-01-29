#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/Policy.hpp>

namespace AIToolbox {
    namespace MDP {
        // Not in header currently
        void updatePolicy(Policy & p, size_t s, const QFunction & q) {
            double max = *std::max_element(std::begin(q[s]), std::end(q[s]));
            std::vector<double> probs(p.getS());
            for ( size_t a = 0; a < p.getA(); ++a ) {
                probs[a] = static_cast<double>(q[s][a] == max);
            }
            p.setStatePolicy(s, probs);
        }
    }
}
