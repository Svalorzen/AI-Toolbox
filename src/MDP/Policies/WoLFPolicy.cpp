#include <AIToolbox/MDP/Policies/WoLFPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::MDP {
    WoLFPolicy::WoLFPolicy(const QFunction & q, const double deltaw, const double deltal, const double scaling) :
            Base(q.rows(), q.cols()), QPolicyInterface(q),
            deltaW_(deltaw), deltaL_(deltal),
            scaling_(scaling), c_(S, 0),
            avgPolicyMatrix_(S,A), actualPolicyMatrix_(S,A),
            avgPolicy_(avgPolicyMatrix_), actualPolicy_(actualPolicyMatrix_)
    {
        avgPolicyMatrix_.fill(1.0/A);
        actualPolicyMatrix_.fill(1.0/A);
    }

    void WoLFPolicy::stepUpdateP(const size_t s) {
        avgPolicyMatrix_.row(s) = avgPolicyMatrix_.row(s) * c_[s] + actualPolicyMatrix_.row(s);
        avgPolicyMatrix_.row(s) /= avgPolicyMatrix_.row(s).sum();
        ++c_[s];

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
            const double avgValue = q_.row(s) * avgPolicyMatrix_.row(s).transpose();
            const double actualValue = q_.row(s) * actualPolicyMatrix_.row(s).transpose();

            auto pickDistribution = std::uniform_int_distribution<unsigned>(0, bestActionCount-1);
            const unsigned selection = pickDistribution(rand_);

            bestAction = bestActions[selection];
            finalDelta = actualValue > avgValue ? deltaW_ : deltaL_;
        }

        finalDelta /= ( c_[s] / scaling_ + 1.0 );

        const auto oldV = actualPolicyMatrix_(s, bestAction);
        actualPolicyMatrix_.row(s) = (actualPolicyMatrix_.row(s).array() - finalDelta/(A-1)).cwiseMax(0);
        actualPolicyMatrix_(s, bestAction) = std::min(1.0, oldV + finalDelta);
        actualPolicyMatrix_.row(s) /= actualPolicyMatrix_.row(s).sum();
    }

    size_t WoLFPolicy::sampleAction(const size_t & s) const {
        return actualPolicy_.sampleAction(s);
    }

    double WoLFPolicy::getActionProbability(const size_t & s, const size_t & a) const {
        return actualPolicy_.getActionProbability(s,a);
    }

    Matrix2D WoLFPolicy::getPolicy() const {
        return actualPolicy_.getPolicy();
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
