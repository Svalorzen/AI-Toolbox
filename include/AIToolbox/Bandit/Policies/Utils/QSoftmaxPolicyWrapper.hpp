#ifndef AI_TOOLBOX_BANDIT_Q_SOFTMAX_POLICY_WRAPPER_HEADER_FILE
#define AI_TOOLBOX_BANDIT_Q_SOFTMAX_POLICY_WRAPPER_HEADER_FILE

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Bandit/Policies/Utils/QGreedyPolicyWrapper.hpp>

namespace AIToolbox::Bandit {
    template <typename V, typename Gen>
    class QSoftmaxPolicyWrapper {
        public:
            QSoftmaxPolicyWrapper(double t, V q, std::vector<size_t> & buffer, Gen & gen);

            size_t sampleAction();
            double getActionProbability(size_t a) const;

            template <typename P>
            void getPolicy(P && p) const;
        private:
            double temperature_;
            V q_;
            std::vector<size_t> & buffer_;
            Gen & rand_;
    };

    // If we get a temporary, we copy it.
    template <typename V, typename Gen>
    QSoftmaxPolicyWrapper(double, const V &&, std::vector<size_t>&, Gen &) -> QSoftmaxPolicyWrapper<V, Gen>;

    // If we get a reference, we store a reference.
    template <typename V, typename Gen>
    QSoftmaxPolicyWrapper(double, const V &, std::vector<size_t>&, Gen &) -> QSoftmaxPolicyWrapper<const V &, Gen>;

    template <typename V, typename Gen>
    QSoftmaxPolicyWrapper<V, Gen>::QSoftmaxPolicyWrapper(double t, V q, std::vector<size_t> & buffer, Gen & gen)
            : temperature_(t), q_(std::move(q)), buffer_(buffer), rand_(gen)
    {
        assert(q_.size() == buffer_.size());
    }

    template <typename V, typename Gen>
    size_t QSoftmaxPolicyWrapper<V, Gen>::sampleAction() {
        if ( checkEqualSmall(temperature_, 0.0) ) {
            auto wrap = QGreedyPolicyWrapper(q_, buffer_, rand_);
            return wrap.sampleAction();
        }

        Vector actionValues = (q_ / temperature_).array().exp();

        unsigned infinities = 0;
        for ( size_t a = 0; a < buffer_.size(); ++a )
            if ( std::isinf(actionValues(a)) )
                buffer_[infinities++] = a;

        if (infinities) {
            auto pickDistribution = std::uniform_int_distribution<unsigned>(0, infinities-1);
            unsigned selection = pickDistribution(rand_);

            return buffer_[selection];
        } else {
            actionValues /= actionValues.sum();

            return sampleProbability(buffer_.size(), actionValues, rand_);
        }
    }

    template <typename V, typename Gen>
    double QSoftmaxPolicyWrapper<V, Gen>::getActionProbability(const size_t a) const {
        if ( checkEqualSmall(temperature_, 0.0) ) {
            auto wrap = QGreedyPolicyWrapper(q_, buffer_, rand_);
            return wrap.getActionProbability(a);
        }

        Vector actionValues = (q_ / temperature_).array().exp();

        bool isAInfinite = false;
        unsigned infinities = 0;
        for ( size_t aa = 0; aa < buffer_.size(); ++aa ) {
            if ( std::isinf(actionValues(aa)) ) {
                infinities++;
                isAInfinite |= (aa == a);
            }
        }
        if ( infinities ) {
            if ( isAInfinite ) return 1.0 / infinities;
            return 0.0;
        }
        return actionValues(a) / actionValues.sum();
    }

    template <typename V, typename Gen>
    template <typename P>
    void QSoftmaxPolicyWrapper<V, Gen>::getPolicy(P && p) const {
        if ( checkEqualSmall(temperature_, 0.0) ) {
            auto wrap = QGreedyPolicyWrapper(q_, buffer_, rand_);
            return wrap.getPolicy(p);
        }

        p = (q_ / temperature_).array().exp();

        unsigned infinities = 0;
        double sum = 0.0;
        for ( size_t a = 0; a < buffer_.size(); ++a ) {
            sum += p[a];
            if ( std::isinf(p[a]) )
                infinities++;
        }

        if ( infinities )
            p = p.array().isInf().template cast<double>() / infinities;
        else if ( checkEqualSmall(sum, 0.0) )
            p.fill(1.0 / buffer_.size());
        else
            p /= sum;
    }
}

#endif
