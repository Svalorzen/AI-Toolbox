#ifndef AI_TOOLBOX_BANDIT_Q_SOFTMAX_POLICY_WRAPPER_HEADER_FILE
#define AI_TOOLBOX_BANDIT_Q_SOFTMAX_POLICY_WRAPPER_HEADER_FILE

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Bandit/Policies/Utils/QGreedyPolicyWrapper.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class implements some basic softmax policy primitives.
     *
     * Since the basic operations on discrete vectors to select an action with
     * softmax are the same both in Bandits and in MDPs, we implement them once
     * here. This class operates on references, so that it does not need to
     * allocate memory and we can keep using the most appropriate storage for
     * whatever problem we are actually working on.
     *
     * Note that you shouldn't really need to specify the template parameters
     * by hand.
     */
    template <typename V, typename Gen>
    class QSoftmaxPolicyWrapper {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param t The temperature to use.
             * @param q A reference to the QFunction to use.
             * @param valueBuffer A buffer to compute softmax values.
             * @param buffer A buffer to determine which action to take in case of equalities.
             * @param gen A random engine.
             */
            QSoftmaxPolicyWrapper(double t, V q, Vector & valueBuffer, std::vector<size_t> & buffer, Gen & gen);

            /**
             * @brief This function chooses an action for state s with probability dependent on value.
             *
             * This class implements softmax through the Boltzmann
             * distribution. Thus an action will be chosen with
             * probability:
             *
             * \f[
             *      P(a) = \frac{e^{(Q(a)/t)})}{\sum_b{e^{(Q(b)/t)}}}
             * \f]
             *
             * where t is the temperature. This value is not cached anywhere, so
             * continuous sampling may not be extremely fast.
             *
             * @return The chosen action.
             */
            size_t sampleAction();

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * \sa sampleAction();
             *
             * @param a The selected action.
             *
             * @return The probability of taking the specified action in the specified state.
             */
            double getActionProbability(size_t a) const;

            /**
             * @brief This function writes in a vector all probabilities of the policy.
             *
             * Ideally this function can be called only when there is a
             * repeated need to access the same policy values in an
             * efficient manner.
             */
            template <typename P>
            void getPolicy(P && p) const;

        private:
            double temperature_;
            V q_;
            Vector & valueBuffer_;
            std::vector<size_t> & buffer_;
            Gen & rand_;
    };

    // If we get a temporary, we copy it.
    template <typename V, typename Gen>
    QSoftmaxPolicyWrapper(double, const V &&, Vector &, std::vector<size_t>&, Gen &) -> QSoftmaxPolicyWrapper<V, Gen>;

    // If we get a reference, we store a reference.
    template <typename V, typename Gen>
    QSoftmaxPolicyWrapper(double, const V &, Vector &, std::vector<size_t>&, Gen &) -> QSoftmaxPolicyWrapper<const V &, Gen>;

    template <typename V, typename Gen>
    QSoftmaxPolicyWrapper<V, Gen>::QSoftmaxPolicyWrapper(double t, V q, Vector & vb, std::vector<size_t> & buffer, Gen & gen)
            : temperature_(t), q_(std::move(q)), valueBuffer_(vb), buffer_(buffer), rand_(gen)
    {
        assert(static_cast<size_t>(q_.size()) == buffer_.size());
    }

    template <typename V, typename Gen>
    size_t QSoftmaxPolicyWrapper<V, Gen>::sampleAction() {
        if ( checkEqualSmall(temperature_, 0.0) ) {
            auto wrap = QGreedyPolicyWrapper(q_, buffer_, rand_);
            return wrap.sampleAction();
        }

        valueBuffer_ = (q_ / temperature_).array().exp();

        unsigned infinities = 0;
        for ( size_t a = 0; a < buffer_.size(); ++a )
            if ( std::isinf(valueBuffer_(a)) )
                buffer_[infinities++] = a;

        if (infinities) {
            auto pickDistribution = std::uniform_int_distribution<unsigned>(0, infinities-1);
            unsigned selection = pickDistribution(rand_);

            return buffer_[selection];
        } else {
            valueBuffer_ /= valueBuffer_.sum();

            return sampleProbability(buffer_.size(), valueBuffer_, rand_);
        }
    }

    template <typename V, typename Gen>
    double QSoftmaxPolicyWrapper<V, Gen>::getActionProbability(const size_t a) const {
        if ( checkEqualSmall(temperature_, 0.0) ) {
            auto wrap = QGreedyPolicyWrapper(q_, buffer_, rand_);
            return wrap.getActionProbability(a);
        }

        valueBuffer_ = (q_ / temperature_).array().exp();

        bool isAInfinite = false;
        unsigned infinities = 0;
        for ( size_t aa = 0; aa < buffer_.size(); ++aa ) {
            if ( std::isinf(valueBuffer_(aa)) ) {
                infinities++;
                isAInfinite |= (aa == a);
            }
        }
        if ( infinities ) {
            if ( isAInfinite ) return 1.0 / infinities;
            return 0.0;
        }
        return valueBuffer_(a) / valueBuffer_.sum();
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
