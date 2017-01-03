#ifndef AI_TOOLBOX_MDP_PRIORITIZED_SWEEPING_EIGEN_HEADER_FILE
#define AI_TOOLBOX_MDP_PRIORITIZED_SWEEPING_EIGEN_HEADER_FILE

#include <tuple>
#include <unordered_map>

#include <boost/heap/fibonacci_heap.hpp>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox {
    namespace MDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_model_eigen<M>::value>::type>
        class PrioritizedSweepingEigen;
#endif

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
        class PrioritizedSweepingEigen<M> {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * @param m The model to be used to update the QFunction.
                 * @param theta The queue threshold.
                 * @param n The number of sampling passes to do on the model upon batchUpdateQ().
                 */
                PrioritizedSweepingEigen(const M & m, double theta = 0.5, unsigned n = 50);

                /**
                 * @brief This function updates the PrioritizedSweepingEigen internal update queue.
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
                 * In PrioritizedSweepingEigen we sample from the queue at most N times for
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

                using PriorityQueueElement = std::tuple<double, size_t>;
                enum {
                    PRIORITY = 0,
                    STATE    = 1,
                };

                class PriorityTupleLess {
                    public:
                        bool operator() (const PriorityQueueElement& arg1, const PriorityQueueElement& arg2) const;
                };

                using QueueType = boost::heap::fibonacci_heap<PriorityQueueElement, boost::heap::compare<PriorityTupleLess>>;

                QueueType queue_;

                std::unordered_map<size_t, typename QueueType::handle_type> queueHandles_;
        };

        template <typename M>
        bool PrioritizedSweepingEigen<M>::PriorityTupleLess::operator() (const PriorityQueueElement& arg1, const PriorityQueueElement& arg2) const
        {
            return std::get<PRIORITY>(arg1) < std::get<PRIORITY>(arg2);
        }

        template <typename M>
        PrioritizedSweepingEigen<M>::PrioritizedSweepingEigen(const M & m, const double theta, const unsigned n) :
                S(m.getS()), A(m.getA()), N(n), theta_(theta), model_(m),
                qfun_(makeQFunction(S,A)), vfun_(makeValueFunction(S)) {}

        template <typename M>
        void PrioritizedSweepingEigen<M>::stepUpdateQ(const size_t s, const size_t a) {
            // We use this to avoid continuous reallocations during the update
            // of q[s][a]
            static Values vector(S);

            auto & values = std::get<VALUES>(vfun_);
            { // Update q[s][a]
                vector.noalias() = values * model_.getDiscount();
                vector += model_.getRewardFunction(a).row(s).transpose();
                qfun_(s,a) = model_.getTransitionFunction(a).row(s).dot(vector);
            }

            double p = values[s];
            {
                // Update value and action
                values[s] = qfun_.row(s).maxCoeff(&std::get<ACTIONS>(vfun_)[s]);
            }

            p = std::fabs(values[s] - p);

            // If it changed enough, we're going to update its parents.
            if ( p > theta_ ) {
                auto it = queueHandles_.find(s);

                if ( it != std::end(queueHandles_) && std::get<PRIORITY>(*(it->second)) < p )
                    queue_.increase(it->second, std::make_tuple(p, s));
                else
                    queueHandles_[s] = queue_.push(std::make_tuple(p, s));
            }
        }

        template <typename M>
        void PrioritizedSweepingEigen<M>::batchUpdateQ() {
            for ( unsigned i = 0; i < N; ++i ) {
                if ( queue_.empty() ) return;

                // The state we extract has been processed already
                // So it is the future we have to backtrack from.
                size_t s1;
                std::tie(std::ignore, s1) = queue_.top();

                queue_.pop();
                queueHandles_.erase(s1);

                for ( size_t s = 0; s < S; ++s )
                    for ( size_t a = 0; a < A; ++a )
                        if ( checkDifferentSmall(model_.getTransitionProbability(s,a,s1), 0.0) )
                            stepUpdateQ(s, a);
            }
        }

        template <typename M>
        void PrioritizedSweepingEigen<M>::setN(const unsigned n) {
            N = n;
        }

        template <typename M>
        unsigned PrioritizedSweepingEigen<M>::getN() const {
            return N;
        }

        template <typename M>
        void PrioritizedSweepingEigen<M>::setQueueThreshold(const double t) {
            if ( t < 0.0 ) throw std::invalid_argument("Theta parameter must be >= 0");
            theta_ = t;
        }

        template <typename M>
        double PrioritizedSweepingEigen<M>::getQueueThreshold() const {
            return theta_;
        }

        template <typename M>
        size_t PrioritizedSweepingEigen<M>::getQueueLength() const {
            return queue_.size();
        }

        template <typename M>
        const M & PrioritizedSweepingEigen<M>::getModel() const {
            return model_;
        }

        template <typename M>
        const QFunction & PrioritizedSweepingEigen<M>::getQFunction() const {
            return qfun_;
        }

        template <typename M>
        void PrioritizedSweepingEigen<M>::setQFunction(const QFunction & qfun) {
            qfun_ = qfun;
        }

        template <typename M>
        const ValueFunction & PrioritizedSweepingEigen<M>::getValueFunction() const {
            return vfun_;
        }
    }
}
#endif
