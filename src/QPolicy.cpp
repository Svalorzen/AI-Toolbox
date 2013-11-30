#include <AIToolbox/MDP/QPolicy.hpp>

#include <AIToolbox/Policy.hpp>

namespace AIToolbox {
    namespace MDP {
        QPolicy::QPolicy(const QFunction & q) : PolicyInterface(q.shape()[0], q.shape()[1]), q_(q) {}

        size_t QPolicy::sampleAction(size_t s) const {
            std::vector<double> probs(S);

            double max = *std::max_element(std::begin(q_[s]), std::end(q_[s]));
            unsigned maxesN = 0;
            for ( size_t a = 0; a < A; a++ ) {
                probs[a] = static_cast<double>(q_[s][a] == max);
                ++maxesN;
            }

            // The multiplication avoids the need to normalize the whole probs vector
            double p = sampleDistribution_(rand_) * maxesN;

            for ( size_t a = 0; a < A; a++ ) {
                if ( probs[a] > p ) return a;
                p -= probs[a];
            }

            // Return last action just in case
            return A-1;
        }

        double QPolicy::getActionProbability(size_t s, size_t a) const {
            double max = *std::max_element(std::begin(q_[s]), std::end(q_[s]));

            if ( q_[s][a] != max ) return 0.0;

            unsigned maxesN = 0;
            for ( size_t a = 0; a < A; a++ ) {
                if (q_[s][a] == max)
                    ++maxesN;
            }
            return 1.0 / maxesN;
        }
    }
}
