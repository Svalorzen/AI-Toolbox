#ifndef AI_TOOLBOX_MDP_PRIORITIZEDSWEEPING_HEADER_FILE
#define AI_TOOLBOX_MDP_PRIORITIZEDSWEEPING_HEADER_FILE

#include <tuple>
#include <unordered_map>

#include <boost/heap/fibonacci_heap.hpp>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace MDP {
        class RLModel;

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
        class PrioritizedSweeping;
#endif

        /**
         * @brief This class represents the PrioritizedSweeping algorithm.
         */
        template <typename M>
        class PrioritizedSweeping<M> {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * @param M The model to be used to update the QFunction.
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
                 *
                 */
                void batchUpdateQ();

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
                 * @param The QFunction that will be copied.
                 */
                void setQFunction(const QFunction &);

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
        bool PrioritizedSweeping<M>::PriorityTupleLess::operator() (const PriorityQueueElement& arg1, const PriorityQueueElement& arg2) const
        {
            return std::get<PRIORITY>(arg1) < std::get<PRIORITY>(arg2);
        }

        template <typename M>
        PrioritizedSweeping<M>::PrioritizedSweeping(const M & m, double theta, unsigned n) :
                                                                                                                S(m.getS()),
                                                                                                                A(m.getA()),
                                                                                                                N(n),
                                                                                                                theta_(theta),
                                                                                                                model_(m),
                                                                                                                qfun_(makeQFunction(S,A)),
                                                                                                                vfun_(makeValueFunction(S)) {}

        template <typename M>
        void PrioritizedSweeping<M>::stepUpdateQ(size_t s, size_t a) {
            auto & values = std::get<VALUES>(vfun_);
            { // Update q[s][a]
                double newQValue = 0;
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    double probability = model_.getTransitionProbability(s,a,s1);
                    if ( checkDifferent( probability, 0.0 ) )
                        newQValue += probability * ( model_.getExpectedReward(s,a,s1) + model_.getDiscount() * values[s1] );
                }
                qfun_[s][a] = newQValue;
            }

            double p = values[s];
            {
                typename QFunction::reference ref = qfun_[s];
                auto begin = std::begin(ref);
                auto it = std::max_element(begin, std::end(ref));
                // Update value and action
                values[s] = *it;
                std::get<ACTIONS>(vfun_)[s] = std::distance(begin, it);
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
                        if ( checkDifferent(model_.getTransitionProbability(s,a,s1), 0.0) )
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
        void PrioritizedSweeping<M>::setQFunction(const QFunction & qfun) {
            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    qfun_[s][a] = qfun[s][a];
        }

        template <typename M>
        const ValueFunction & PrioritizedSweeping<M>::getValueFunction() const {
            return vfun_;
        }
    }
}
#endif
