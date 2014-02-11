#ifndef AI_TOOLBOX_MDP_PRIORITIZEDSWEEPING_HEADER_FILE
#define AI_TOOLBOX_MDP_PRIORITIZEDSWEEPING_HEADER_FILE

#include <tuple>
#include <unordered_map>

#include <boost/heap/fibonacci_heap.hpp>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox {
    namespace MDP {
        class RLModel;
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
        class PrioritizedSweeping;

        /**
         * @brief This class represents the PrioritizedSweeping algorithm.
         */
        template <typename M>
        class PrioritizedSweeping<M> {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param alpha The learning rate of the QLearning method.
                 * @param discount The discount of the QLearning method.
                 * @param theta The queue threshold.
                 * @param n The number of sampling passes to do on the model upon batchUpdateQ().
                 */
                PrioritizedSweeping(const M & m, double discount = 0.9, double theta = 0.5, unsigned n = 50);

                /**
                 * @brief This function updates the PrioritizedSweeping internal update queue.
                 *
                 * Note that this function does NOT update the QFunction yet, but instead
                 * waits for the batchUpdateQ() call before doing that.
                 *
                 * @param s The previous state.
                 * @param s1 The new state.
                 * @param a The action performed.
                 * @param rew The reward obtained.
                 * @param q A pointer to the QFunction that is begin accessed.
                 */
                void stepUpdateQ(size_t s, size_t a);

                /**
                 * @brief This function updates a QFunction based on simulated experience.
                 *
                 * In PrioritizedSweeping we sample from the queue at most N times for
                 * state action pairs that need updating. For each one of them we update
                 * the QFunction and recursively check whether this produces new changes
                 * worth updating. If so, they are inserted in the queue_ and the function
                 * proceeds to the nest iteration.
                 *
                 * @param m The RLModel we sample experience from.
                 * @param q The QFunction to update.
                 */
                void batchUpdateQ();

                /**
                 * @brief This function sets the discount parameter.
                 *
                 * The discount parameter must be > 0.0 and <= 1.0,
                 * otherwise the function will throw an std::invalid_argument.
                 *
                 * @param d The new discount parameter.
                 */
                void setDiscount(double d);

                /**
                 * @brief This function will return the currently set discount parameter.
                 *
                 * @return The currently set discount parameter.
                 */
                double getDiscount() const;

                /**
                 * @brief This function sets the theta parameter.
                 *
                 * The discount parameter must be >= 0.0.
                 * otherwise the function will throw an std::invalid_argument.
                 *
                 * @param d The new theta parameter.
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

                size_t getQueueLength() const;
                const M & getModel() const;
                const QFunction & getQFunction() const;
                const ValueFunction & getValueFunction() const;

            private:
                size_t S, A;
                unsigned N;
                double discount_, theta_;

                const M & model_;
                QFunction qfun_;
                ValueFunction vfun_;

                using PriorityQueueElement = std::tuple<double, size_t>;

                class PriorityTupleLess {
                    public:
                        bool operator() (const PriorityQueueElement& arg1, const PriorityQueueElement& arg2) const;
                };

                using QueueType = boost::heap::fibonacci_heap<PriorityQueueElement, boost::heap::compare<PriorityTupleLess>>;

                QueueType queue_;

                std::unordered_map<size_t, typename QueueType::handle_type> queueHandles_;
        };

        template <typename M>
        bool PrioritizedSweeping<M>::PriorityTupleLess::operator() (const PriorityQueueElement& arg1, const PriorityQueueElement& arg2) const
        {
            return std::get<0>(arg1) < std::get<0>(arg2);
        }

        template <typename M>
        PrioritizedSweeping<M>::PrioritizedSweeping(const M & m, double discount, double theta, unsigned n) :
                                                                                                                S(m.getS()),
                                                                                                                A(m.getA()),
                                                                                                                N(n),
                                                                                                                discount_(discount),
                                                                                                                theta_(theta),
                                                                                                                model_(m),
                                                                                                                qfun_(makeQFunction(S,A)),
                                                                                                                vfun_(S, 0.0)
        {
            if ( discount <= 0.0 || discount > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        }

        template <typename M>
        void PrioritizedSweeping<M>::stepUpdateQ(size_t s, size_t a) {
            { // Update q[s][a]
                double newQValue = 0;
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    double probability = model_.getTransitionProbability(s, s1, a);
                    if ( probability > 0 )
                        newQValue += probability * ( model_.getExpectedReward(s, s1, a) + discount_ * vfun_[s1] );
                }
                qfun_[s][a] = newQValue;
            }

            double p = vfun_[s];
            vfun_[s] = *std::max_element(std::begin(qfun_[s]), std::end(qfun_[s]));

            p = std::fabs(vfun_[s] - p);

            // If it changed enough, we're going to update its parents.
            if ( p > theta_ ) {
                auto it = queueHandles_.find(s);

                if ( it != std::end(queueHandles_) && std::get<0>(*(it->second)) < p )
                    queue_.increase(it->second, std::make_tuple(p, s));
                else
                    queueHandles_[s] = queue_.push(std::make_tuple(p, s));
            }
        }

        template <typename M>
        void PrioritizedSweeping<M>::batchUpdateQ() {
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
                        if ( model_.getTransitionProbability(s, s1, a) > 0.0 )
                            stepUpdateQ(s, a);
            }
        }

        template <typename M>
        void PrioritizedSweeping<M>::setN(unsigned n) {
            N = n;
        }

        template <typename M>
        unsigned PrioritizedSweeping<M>::getN() const {
            return N;
        }

        template <typename M>
        void PrioritizedSweeping<M>::setDiscount(double d) {
            if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
            discount_ = d;
        }

        template <typename M>
        double PrioritizedSweeping<M>::getDiscount() const {
            return discount_;
        }

        template <typename M>
        void PrioritizedSweeping<M>::setQueueThreshold(double t) {
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
        const ValueFunction & PrioritizedSweeping<M>::getValueFunction() const {
            return vfun_;
        }
    }
}
#endif
