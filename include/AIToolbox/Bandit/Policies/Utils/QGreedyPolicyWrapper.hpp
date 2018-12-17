#ifndef AI_TOOLBOX_BANDIT_Q_GREEDY_POLICY_WRAPPER_HEADER_FILE
#define AI_TOOLBOX_BANDIT_Q_GREEDY_POLICY_WRAPPER_HEADER_FILE

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Bandit/Types.hpp>

namespace AIToolbox::Bandit {
    template <typename V, typename Gen>
    class QGreedyPolicyWrapper {
        public:
            QGreedyPolicyWrapper(V q, std::vector<size_t> & buffer, Gen & gen);

            size_t sampleAction();
            double getActionProbability(size_t a) const;

            template <typename P>
            void getPolicy(P && p) const;

        private:
            V q_;
            std::vector<size_t> & buffer_;
            Gen & rand_;
    };

    template <typename V, typename Gen>
    QGreedyPolicyWrapper(const V &&, std::vector<size_t>&, Gen &) -> QGreedyPolicyWrapper<V, Gen>;

    template <typename V, typename Gen>
    QGreedyPolicyWrapper(const V &, std::vector<size_t>&, Gen &) -> QGreedyPolicyWrapper<const V &, Gen>;

    template <typename V, typename Gen>
    QGreedyPolicyWrapper<V, Gen>::QGreedyPolicyWrapper(V q, std::vector<size_t> & buffer, Gen & gen)
            : q_(std::move(q)), buffer_(buffer), rand_(gen)
    {
        assert(q_.size() == buffer_.size());
    }

    template <typename V, typename Gen>
    size_t QGreedyPolicyWrapper<V, Gen>::sampleAction() {
        // Automatically sets initial best action as bestAction[0] = 0
        buffer_[0] = 0;

        // This work is due to multiple max-valued actions
        double bestValue = q_[0]; unsigned bestActionCount = 1;
        for ( size_t a = 1; a < buffer_.size(); ++a ) {
            const double val = q_[a];
            // The checkEqualGeneral is before the greater since we want to
            // trap here things that may be equal (even if one is a tiny bit
            // higher than the other).
            if ( checkEqualGeneral(val, bestValue) ) {
                buffer_[bestActionCount] = a;
                ++bestActionCount;
            }
            else if ( val > bestValue ) {
                buffer_[0] = a;
                bestActionCount = 1;
                bestValue = val;
            }
        }
        auto pickDistribution = std::uniform_int_distribution<unsigned>(0, bestActionCount-1);
        const unsigned selection = pickDistribution(rand_);

        return buffer_[selection];
    }

    template <typename V, typename Gen>
    double QGreedyPolicyWrapper<V, Gen>::getActionProbability(const size_t a) const {
        const double max = q_[a]; unsigned count = 0;
        for ( size_t aa = 0; aa < buffer_.size(); ++aa ) {
            const double val = q_[aa];
            // The checkEqualGeneral is before the greater since we want to
            // trap here things that may be equal (even if one is a tiny bit
            // higher than the other).
            if ( checkEqualGeneral(val, max) ) ++count;
            else if ( val > max ) {
                return 0.0;
            }
        }
        return 1.0 / count;
    }

    template <typename V, typename Gen>
    template <typename P>
    void QGreedyPolicyWrapper<V, Gen>::getPolicy(P && p) const {
        double max = q_[0]; unsigned count = 1;
        for ( size_t aa = 1; aa < buffer_.size(); ++aa ) {
            const double val = q_[aa];
            // The checkEqualGeneral is before the greater since we want to
            // trap here things that may be equal (even if one is a tiny bit
            // higher than the other).
            if ( checkEqualGeneral(val, max) ) ++count;
            else if ( val > max ) {
                max = val;
                count = 1;
            }
        }
        for ( size_t aa = 0; aa < buffer_.size(); ++aa ) {
            if ( checkEqualGeneral(q_[aa], max) )
                p[aa] = 1.0 / count;
            else
                p[aa] = 0.0;
        }
    }
};

#endif
