#ifndef AI_TOOLBOX_MDP_PRIORITIZED_SWEEPING_HEADER_FILE
#define AI_TOOLBOX_MDP_PRIORITIZED_SWEEPING_HEADER_FILE

#include <tuple>
#include <unordered_map>

#include <boost/heap/fibonacci_heap.hpp>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the PrioritizedSweeping algorithm.
     *
     * This algorithm is a refinement of the DynaQ algorithm. Instead of
     * randomly sampling experienced state action pairs to get more
     * information, we order each pair based on an estimate of how much
     * information we can still extract from them.
     *
     * In particular, pairs are sorted based on the amount they modified
     * the estimated ValueFunction on their last sample. This ensures that
     * we always try to sample from useful pairs instead of randomly,
     * extracting knowledge much faster.
     *
     * At the same time, this algorithm keeps a threshold for each
     * state-action pair, so that it does not have to internally store all
     * the pairs and save some memory/cpu time keeping the queue updated.
     * Only pairs which obtained an amount of change higher than this
     * treshold are kept in the queue.
     *
     * Differently from the QLearning and DynaQ algorithm, this class
     * automatically computes the ValueFunction since it is useful to
     * determine which state-action pairs are actually useful, so there's
     * no need to compute it manually.
     *
     * Given how this algorithm updates the QFunction, the only problems
     * supported by this approach are ones with an infinite horizon.
     */
    template <typename M>
    class PrioritizedSweeping {
        static_assert(is_model_v<M>, "This class only works for MDP models!");

        public:
            /**
             * @brief Basic constructor.
             *
             * @param m The model to be used to update the QFunction.
             * @param theta The queue threshold.
             * @param n The number of sampling passes to do on the model upon batchUpdateQ().
             */
            PrioritizedSweeping(const M & m, double theta = 0.5, unsigned n = 50);

            /**
             * @brief This function updates the PrioritizedSweeping internal update queue.
             *
             * This function updates the QFunction for the specified pair, and decides
             * whether any parent couple that can lead to this state is worth pushing
             * into the queue.
             *
             * @param s The previous state.
             * @param a The action performed.
             */
            void stepUpdateQ(size_t s, size_t a);

            /**
             * @brief This function updates a QFunction based on simulated experience.
             *
             * In PrioritizedSweeping we sample from the queue at most N times for
             * state action pairs that need updating. For each one of them we update
             * the QFunction and recursively check whether this produces new changes
             * worth updating. If so, they are inserted in the queue_ and the function
             * proceeds to the next most urgent iteration.
             */
            void batchUpdateQ();

            /**
             * @brief This function sets the theta parameter.
             *
             * The discount parameter must be >= 0.0.
             * otherwise the function will throw an std::invalid_argument.
             *
             * @param t The new theta parameter.
             */
            void setQueueThreshold(double t);

            /**
             * @brief This function will return the currently set theta parameter.
             *
             * @return The currently set theta parameter.
             */
            double getQueueThreshold() const;

            /**
             * @brief This function sets the number of sampling passes during batchUpdateQ().
             *
             * @param n The new number of updates.
             */
            void setN(unsigned n);

            /**
             * @brief This function returns the currently set number of sampling passes during batchUpdateQ().
             *
             * @return The current number of updates().
             */
            unsigned getN() const;

            /**
             * @brief This function returns the current number of elements unprocessed in the queue.
             *
             * @return The current length of the queue.
             */
            size_t getQueueLength() const;

            /**
             * @brief This function returns a reference to the referenced Model.
             *
             * @return The internal Model.
             */
            const M & getModel() const;

            /**
             * @brief This function returns a reference to the internal QFunction.
             *
             * @return The internal QFunction.
             */
            const QFunction & getQFunction() const;

            /**
             * @brief This function allows you to set the value of the internal QFunction.
             *
             * This function can be useful in case you are starting with an already populated
             * Experience/Model, which you can solve (for example with ValueIteration)
             * and then improve the solution with new experience.
             *
             * @param q The QFunction that will be copied.
             */
            void setQFunction(const QFunction & q);

            /**
             * @brief This function returns a reference to the internal ValueFunction.
             *
             * @return The internal ValueFunction.
             */
            const ValueFunction & getValueFunction() const;

        private:
            size_t S, A;
            unsigned N;
            double theta_;

            const M & model_;
            QFunction qfun_;
            ValueFunction vfun_;

            struct PriorityQueueElement {
                double priority;
                std::pair<size_t, size_t> stateAction;
                bool operator<(const PriorityQueueElement& arg2) const {
                    return priority < arg2.priority;
                }
            };

            using QueueType = boost::heap::fibonacci_heap<PriorityQueueElement>;

            QueueType queue_;

            std::unordered_map<std::pair<size_t, size_t>, typename QueueType::handle_type, boost::hash<std::pair<size_t, size_t>>> queueHandles_;
    };

    template <typename M>
    PrioritizedSweeping<M>::PrioritizedSweeping(const M & m, const double theta, const unsigned n) :
            S(m.getS()), A(m.getA()), N(n), theta_(theta), model_(m),
            qfun_(makeQFunction(S,A)), vfun_(makeValueFunction(S)) {}

    template <typename M>
    void PrioritizedSweeping<M>::stepUpdateQ(const size_t s, const size_t a) {
        auto & values = vfun_.values;

        // Update q[s][a]
        if constexpr(is_model_eigen_v<M>) {
            qfun_(s,a) = model_.getRewardFunction().coeff(s, a) + model_.getTransitionFunction(a).row(s).dot(values * model_.getDiscount());
        } else {
            double newQValue = 0;
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                const double probability = model_.getTransitionProbability(s,a,s1);
                if ( checkDifferentSmall( probability, 0.0 ) )
                    newQValue += probability * ( model_.getExpectedReward(s,a,s1) + model_.getDiscount() * values[s1] );
            }
            qfun_(s, a) = newQValue;
        }

        double p = values[s];
        {
            // Update value and action
            values[s] = qfun_.row(s).maxCoeff(&(vfun_.actions[s]));
        }

        p = std::fabs(values[s] - p);

        for ( size_t ss = 0; ss < S; ++ss ) {
            for ( size_t a = 0; a < A; ++a ) {
                const double delta = p * model_.getTransitionProbability(ss,a,s);
                // If it changed enough, we're going to update its parents.
                if ( delta > theta_ ) {
                    const auto pair = std::make_pair(ss, a);
                    auto it = queueHandles_.find(pair);

                    if (it != std::end(queueHandles_)) {
                        if ((*it->second).priority < delta) {
                            (*it->second).priority = delta;
                            queue_.increase(it->second);
                        }
                    } else {
                        queueHandles_[pair] = queue_.emplace(PriorityQueueElement{delta, pair});
                    }
                }
            }
        }
    }

    template <typename M>
    void PrioritizedSweeping<M>::batchUpdateQ() {
        for ( unsigned i = 0; i < N; ++i ) {
            if ( queue_.empty() ) return;

            // The state we extract has been processed already
            // So it is the future we have to backtrack from.
            auto [p, pair] = queue_.top();
            (void)p;

            queue_.pop();
            queueHandles_.erase(pair);

            stepUpdateQ(pair.first, pair.second);
        }
    }

    template <typename M>
    void PrioritizedSweeping<M>::setN(const unsigned n) {
        N = n;
    }

    template <typename M>
    unsigned PrioritizedSweeping<M>::getN() const {
        return N;
    }

    template <typename M>
    void PrioritizedSweeping<M>::setQueueThreshold(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Theta parameter must be >= 0");
        theta_ = t;
    }

    template <typename M>
    double PrioritizedSweeping<M>::getQueueThreshold() const {
        return theta_;
    }

    template <typename M>
    size_t PrioritizedSweeping<M>::getQueueLength() const {
        return queue_.size();
    }

    template <typename M>
    const M & PrioritizedSweeping<M>::getModel() const {
        return model_;
    }

    template <typename M>
    const QFunction & PrioritizedSweeping<M>::getQFunction() const {
        return qfun_;
    }

    template <typename M>
    void PrioritizedSweeping<M>::setQFunction(const QFunction & qfun) {
        qfun_ = qfun;
    }

    template <typename M>
    const ValueFunction & PrioritizedSweeping<M>::getValueFunction() const {
        return vfun_;
    }
}

#endif
