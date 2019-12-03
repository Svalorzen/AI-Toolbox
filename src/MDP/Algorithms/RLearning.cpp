#include <AIToolbox/MDP/Algorithms/RLearning.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::MDP {
    RLearning::RLearning(const size_t ss, const size_t aa, const double alpha, const double rho) :
            S(ss), A(aa), rAvg_(0.0), q_(makeQFunction(S, A))
    {
        setAlphaLearningRate(alpha);
        setRhoLearningRate(rho);
    }

    void RLearning::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const double rew) {
        const double futureBestValue = q_.row(s1).maxCoeff();
        q_(s, a) += alpha_ * ( rew - rAvg_ + futureBestValue );

        const double currBestValue = q_.row(s).maxCoeff();
        if (checkEqualGeneral(q_(s, a), currBestValue))
            rAvg_ += rho_ * ( rew + futureBestValue - currBestValue );
    }

    void RLearning::setAlphaLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Alpha learning rate parameter must be in (0,1]");
        alpha_ = a;
    }


    void RLearning::setRhoLearningRate(const double r) {
        if ( r <= 0.0 || r > 1.0 ) throw std::invalid_argument("Rho learning rate parameter must be in (0,1]");
        rho_ = r;
    }

    double RLearning::getAlphaLearningRate() const { return alpha_; }
    double RLearning::getRhoLearningRate() const { return rho_; }

    size_t RLearning::getS() const { return S; }
    size_t RLearning::getA() const { return A; }

    const QFunction & RLearning::getQFunction() const { return q_; }
    double RLearning::getAverageReward() const { return rAvg_; }

    void RLearning::setQFunction(const QFunction & qfun) {
        assert(q_.rows() == qfun.rows());
        assert(q_.cols() == qfun.cols());
        q_ = qfun;
    }
}

