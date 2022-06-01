#include <AIToolbox/Bandit/Policies/SuccessiveRejectsPolicy.hpp>

namespace AIToolbox::Bandit {
    SuccessiveRejectsPolicy::SuccessiveRejectsPolicy(const Experience & exp, unsigned budget) :
            Base(exp.getA()), exp_(exp), budget_(budget),
            currentPhase_(1), currentActionId_(0), currentArmPulls_(0),
            nKOld_(0), nKNew_(0), logBarK_(0.5), availableActions_(getA())
    {
        for (unsigned i = 2; i <= getA(); ++i)
            logBarK_ += 1.0 / i;

        updateNks();

        std::iota(std::begin(availableActions_), std::end(availableActions_), 0);
    }

    size_t SuccessiveRejectsPolicy::sampleAction() const {
        return availableActions_[currentActionId_];
    }

    void SuccessiveRejectsPolicy::stepUpdateQ() {
        ++currentArmPulls_;

        if (currentArmPulls_ < (nKNew_ - nKOld_))
            return;

        currentArmPulls_ = 0;
        ++currentActionId_;

        if (currentActionId_ < availableActions_.size())
            return;

        currentActionId_ = 0;
        ++currentPhase_;

        if (currentPhase_ > getA())
            return;

        updateNks();

        // Remove worst action from selection pool
        size_t minAction = availableActions_[0];
        double minValue = exp_.getRewardMatrix()[minAction];

        for (size_t i = 1; i < availableActions_.size(); ++i) {
            const size_t a = availableActions_[i];
            const double v = exp_.getRewardMatrix()[a];

            if (v < minValue) {
                minValue = v;
                minAction = a;
            }
        }

        auto it = std::find(std::begin(availableActions_), std::end(availableActions_), minAction);
        *it = availableActions_.back();
        availableActions_.pop_back();
    }

    bool SuccessiveRejectsPolicy::canRecommendAction() const {
        return availableActions_.size() == 1;
    }

    size_t SuccessiveRejectsPolicy::recommendAction() const {
        return availableActions_[0];
    }

    void SuccessiveRejectsPolicy::updateNks() {
        nKOld_ = nKNew_;
        nKNew_ = std::ceil(
            ( budget_ - getA() ) / (logBarK_ * (getA() + 1 - currentPhase_) )
        );
    }

    size_t SuccessiveRejectsPolicy::getCurrentPhase() const { return currentPhase_; }
    size_t SuccessiveRejectsPolicy::getCurrentNk() const { return nKNew_; }

    double SuccessiveRejectsPolicy::getActionProbability(const size_t & a) const { return a == availableActions_[currentActionId_]; }
    Vector SuccessiveRejectsPolicy::getPolicy() const {
        Vector v(getA());
        v.setZero();

        v[availableActions_[currentActionId_]] = 1.0;

        return v;
    }

    const Experience & SuccessiveRejectsPolicy::getExperience() const {
        return exp_;
    }
}
