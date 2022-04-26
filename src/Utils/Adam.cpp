#include <AIToolbox/Utils/Adam.hpp>

namespace AIToolbox {
    Adam::Adam(AIToolbox::Vector * point, const AIToolbox::Vector & gradient, const double alpha, const double beta1, const double beta2, const double epsilon) :
        point_(point), gradient_(&gradient),
        m_(point_->size()), v_(point_->size()),
        beta1_(beta1), beta2_(beta2), alpha_(alpha), epsilon_(epsilon),
        step_(1)
    {
        reset();
    }

    void Adam::step() {
        assert(point_);
        assert(gradient_);

        m_ = beta1_ * m_ + (1.0 - beta1_) * (*gradient_);
        v_ = beta2_ * v_ + (1.0 - beta2_) * (*gradient_).array().square().matrix();

        const double alphaHat = alpha_ * std::sqrt(1.0 - std::pow(beta2_, step_)) / (1.0 - std::pow(beta1_, step_));

        (*point_).array() -= alphaHat * m_.array() / (v_.array().sqrt() + epsilon_);

        ++step_;
    }

    void Adam::reset() {
        m_.fill(0.0);
        v_.fill(0.0);
        step_ = 1;
    }

    void Adam::reset(AIToolbox::Vector * point, const AIToolbox::Vector & gradient) {
        point_ = point;
        gradient_ = &gradient;
        reset();
    }

    void Adam::setBeta1(double beta1) { beta1_ = beta1; }
    void Adam::setBeta2(double beta2) { beta2_ = beta2; }
    void Adam::setAlpha(double alpha) { alpha_ = alpha; }
    void Adam::setEpsilon(double epsilon) { epsilon_ = epsilon; }

    double Adam::getBeta1() const { return beta1_; }
    double Adam::getBeta2() const { return beta2_; }
    double Adam::getAlpha() const { return alpha_; }
    double Adam::getEpsilon() const { return epsilon_; }
}
