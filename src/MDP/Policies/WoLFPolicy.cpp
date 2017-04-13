#include <AIToolbox/MDP/Policies/WoLFPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox {
    namespace MDP {

        WoLFPolicy::WoLFPolicy(const QFunction & q, const double deltaw, const double deltal, const double scaling) :
                QPolicyInterface(q), deltaW_(deltaw), deltaL_(deltal),
                scaling_(scaling), c_(S, 0), avgPolicy_(S,A), actualPolicy_(S,A) {}

        void WoLFPolicy::updatePolicy(const size_t s) {
            ++c_[s];

            auto avgstate = avgPolicy_.getStatePolicy(s);
            // Obtain argmax of Q[s], check whether we are losing or winning.
            auto actualstate = actualPolicy_.getStatePolicy(s);

            // Update estimate of average policy
            avgstate.noalias() += (1.0/c_[s]) * (actualstate - avgstate);
            avgstate /= avgstate.sum();
            avgPolicy_.setStatePolicy(s, avgstate);

            size_t bestAction; double finalDelta;
            {
                unsigned bestActionCount = 1; double bestQValue = q_(s, 0);
                // Automatically sets initial best action as bestAction[0] = 0
                std::vector<size_t> bestActions(A,0);

                for ( size_t a = 1; a < A; ++a ) {
                    const double qsa = q_(s, a);

                    if ( checkEqualGeneral(qsa, bestQValue) ) {
                        bestActions[bestActionCount] = a;
                        ++bestActionCount;
                    }
                    else if ( qsa > bestQValue ) {
                        bestActions[0] = a;
                        bestActionCount = 1;
                        bestQValue = qsa;
                    }
                }
                double avgValue = q_.row(s) * avgstate;
                double actualValue = q_.row(s) * actualstate;

                auto pickDistribution = std::uniform_int_distribution<unsigned>(0, bestActionCount-1);
                const unsigned selection = pickDistribution(rand_);

                bestAction = bestActions[selection];
                finalDelta = actualValue > avgValue ? deltaW_ : deltaL_;
            }

            finalDelta /= ( c_[s] / scaling_ + 1.0 );

            auto oldV = actualstate(bestAction);
            actualstate = (actualstate.array() - finalDelta/(A-1)).cwiseMax(0);
            actualstate(bestAction) = std::min(1.0, oldV + finalDelta);
            actualstate /= actualstate.sum();

            // Policy automatically normalizes this to 1
            actualPolicy_.setStatePolicy(s, actualstate);
        }

        size_t WoLFPolicy::sampleAction(const size_t & s) const {
            return actualPolicy_.sampleAction(s);
        }

        double WoLFPolicy::getActionProbability(const size_t & s, const size_t & a) const {
            return actualPolicy_.getActionProbability(s,a);
        }

        void WoLFPolicy::setDeltaW(const double deltaW) {
            deltaW_ = deltaW;
        }

        double WoLFPolicy::getDeltaW() const {
            return deltaW_;
        }

        void WoLFPolicy::setDeltaL(const double deltaL) {
            deltaL_ = deltaL;
        }

        double WoLFPolicy::getDeltaL() const {
            return deltaL_;
        }

        void WoLFPolicy::setScaling(const double scaling) {
            scaling_ = scaling;
        }

        double WoLFPolicy::getScaling() const {
            return scaling_;
        }
    }
}
