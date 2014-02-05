#include <AIToolbox/MDP/PrioritizedSweeping.hpp>

#include <string>

#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox {
    namespace MDP {
        bool PrioritizedSweeping::PriorityTupleLess::operator() (const PriorityQueueElement& arg1, const PriorityQueueElement& arg2) const
        {
            return std::get<0>(arg1) < std::get<0>(arg2);
        }

        PrioritizedSweeping::PrioritizedSweeping(const RLModel & m, double discount, double theta, unsigned n) : S(m.getS()), A(m.getA()), N(n), discount_(discount), theta_(theta), model_(m), qfun_(makeQFunction(S,A)), vfun_(S, 0.0) {}

        void PrioritizedSweeping::stepUpdateQ(size_t s, size_t a) {
            { // Update q[s][a]
                double newQValue = 0;
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    double probability = model_.getTransitionProbability(s, s1, a);
                    if ( probability > 0 )
                        newQValue += probability * ( model_.getExpectedReward(s, s1, a) + discount_ * vfun_[s1] );
                }
                qfun_[s][a] = newQValue;
            }

            double p = vfun_[s];
            vfun_[s] = *std::max_element(std::begin(qfun_[s]), std::end(qfun_[s]));

            p = std::fabs(vfun_[s] - p);

            // If it changed enough, we're going to update its parents.
            if ( p > theta_ ) {
                auto it = queueHandles_.find(s);

                if ( it != std::end(queueHandles_) && std::get<0>(*(it->second)) < p )
                    queue_.increase(it->second, std::make_tuple(p, s));
                else
                    queueHandles_[s] = queue_.push(std::make_tuple(p, s));
            }
        }

        void PrioritizedSweeping::batchUpdateQ() {
            for ( unsigned i = 0; i < N; ++i ) {
                if ( queue_.empty() ) return;

                // The state we extract has been processed already
                // So it is the future we have to backtrack from.
                size_t s1;
                std::tie(std::ignore, s1) = queue_.top();

                queue_.pop();
                queueHandles_.erase(s1);

                for ( size_t s = 0; s < S; ++s )
                    for ( size_t a = 0; a < A; ++a )
                        if ( model_.getTransitionProbability(s, s1, a) > 0.0 )
                            stepUpdateQ(s, a);
            }
        }

        size_t PrioritizedSweeping::getQueueLength() const {
            return queue_.size();
        }

        const RLModel & PrioritizedSweeping::getModel() const {
            return model_;
        }

        const QFunction & PrioritizedSweeping::getQFunction() const {
            return qfun_;
        }

        const ValueFunction & PrioritizedSweeping::getValueFunction() const {
            return vfun_;
        }
    }
}
