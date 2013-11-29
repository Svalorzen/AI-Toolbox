#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/Policy.hpp>

namespace AIToolbox {
    namespace MDP {
        Policy makePolicy(size_t S, size_t A, const QFunction & q) {
            Policy p(S,A);
            std::vector<double> probs(S);
            for ( size_t s = 0; s < S; s++ ) {
                double max = *std::max_element(std::begin(q[s]), std::end(q[s]));
                for ( size_t a = 0; a < A; a++ ) {
                    probs[a] = static_cast<double>(q[s][a] == max);
                }
                p.setPolicy(s, probs);
            }
            return p;
        }

        void updatePolicy(Policy & p, size_t s, const QFunction & q) {
            double max = *std::max_element(std::begin(q[s]), std::end(q[s]));
            std::vector<double> probs(p.getS());
            for ( size_t a = 0; a < p.getA(); a++ ) {
                probs[a] = static_cast<double>(q[s][a] == max);
            }
            p.setPolicy(s, probs);
        }
    }
}
