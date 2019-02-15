#include <AIToolbox/MDP/Policies/PGAAPPPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::MDP {
    PGAAPPPolicy::PGAAPPPolicy(const QFunction & q, const double lRate, const double predictionLength) :
            Base(q.rows(), q.cols()), QPolicyInterface(q),
            lRate_(lRate), predictionLength_(predictionLength),
            policyMatrix_(S,A), policy_(policyMatrix_)
    {
        if ( lRate_ < 0.0 ) throw std::invalid_argument("Learning rate must be >= 0");
        if ( predictionLength_ < 0.0 ) throw std::invalid_argument("Prediction length must be >= 0");

        policyMatrix_.fill(1.0/A);
    }

    void PGAAPPPolicy::stepUpdateP(const size_t s) {
        const double avgR = policyMatrix_.row(s) * q_.row(s).transpose();

        for (size_t a = 0; a < A; ++a) {
            double delta;
            if (checkEqualSmall(policyMatrix_(s, a), 1.0))
                delta = q_(s, a) - avgR;
            else
                delta = (q_(s, a) - avgR) / (1.0 - policyMatrix_(s, a));

            delta -= predictionLength_ * policyMatrix_(s, a) * std::fabs(delta);
            policyMatrix_(s, a) += lRate_ * delta;
        }

        policyMatrix_.row(s) = projectToProbability(policyMatrix_.row(s));
    }

    size_t PGAAPPPolicy::sampleAction(const size_t & s) const {
        return policy_.sampleAction(s);
    }

    double PGAAPPPolicy::getActionProbability(const size_t & s, const size_t & a) const {
        return policy_.getActionProbability(s,a);
    }

    Matrix2D PGAAPPPolicy::getPolicy() const {
        return policy_.getPolicy();
    }

    void PGAAPPPolicy::setLearningRate(const double lRate) {
        if ( lRate < 0.0 ) throw std::invalid_argument("Learning rate must be >= 0");
        lRate_ = lRate;
    }

    double PGAAPPPolicy::getLearningRate() const {
        return lRate_;
    }

    void PGAAPPPolicy::setPredictionLength(const double pLength) {
        if ( pLength < 0.0 ) throw std::invalid_argument("Prediction length must be >= 0");
        predictionLength_ = pLength;
    }

    double PGAAPPPolicy::getPredictionLength() const {
        return predictionLength_;
    }
}
