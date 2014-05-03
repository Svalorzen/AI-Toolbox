#ifndef AI_TOOLBOX_MDP_DYNAQ_HEADER_FILE
#define AI_TOOLBOX_MDP_DYNAQ_HEADER_FILE

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/QLearning.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

#include <boost/functional/hash.hpp>
#include <utility>
#include <unordered_set>
#include <vector>

namespace AIToolbox {
    namespace MDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
        class DynaQ;
#endif
        /**
         * @brief This class represents the DynaQ algorithm.
         */
        template <typename M>
        class DynaQ<M> {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * @param M The model to be used to update the QFunction.
                 * @param alpha The learning rate of the QLearning method.
                 * @param n The number of sampling passes to do on the model upon batchUpdateQ().
                 */
                explicit DynaQ(const M & m, double alpha = 0.5, unsigned n = 50);

                /**
                 * @brief This function updates the internal QFunction.
                 *
                 * This function takes a single experience point and uses it to update
                 * a QFunction. This is a very efficient method to keep the QFunction
                 * up to date with the latest experience.
                 *
                 * In addition, the sampling list is updated so that batch
                 * updating becomes possible as a second phase.
                 *
                 * The sampling list in DynaQ is a simple list of all visited
                 * state action pairs. This function is responsible for inserting
                 * them in a set, keeping them unique.
                 *
                 * @param s The previous state.
                 * @param s1 The new state.
                 * @param a The action performed.
                 * @param rew The reward obtained.
                 */
                void stepUpdateQ(size_t s, size_t s1, size_t a, double rew);

                /**
                 * @brief This function updates a QFunction based on simulated experience.
                 *
                 * In DynaQ we sample N times from already experienced state-action pairs,
                 * and we update the resulting QFunction as if this experience was actually
                 * real.
                 *
                 */
                void batchUpdateQ();

                /**
                 * @brief This function sets the learning rate parameter.
                 *
                 * The learning rate parameter must be > 0.0 and <= 1.0,
                 * otherwise the function will throw an std::invalid_argument.
                 *
                 * @param a The new learning rate parameter.
                 */
                void setLearningRate(double a);

                /**
                 * @brief This function will return the current set learning rate parameter.
                 *
                 * @return The currently set learning rate parameter.
                 */
                double getLearningRate() const;

                /**
                 * @brief This function sets the current sample number parameter.
                 *
                 * @param n The new sample number parameter.
                 */
                void setN(unsigned n);

                /**
                 * @brief This function returns the number of states of the world.
                 *
                 * @return The total number of states.
                 */
                size_t getS() const;

                /**
                 * @brief This function returns the number of available actions to the agent.
                 *
                 * @return The total number of actions.
                 */
                size_t getA() const;

                /**
                 * @brief This function returns the currently set number of sampling passes during batchUpdateQ().
                 *
                 * @return The current number of updates().
                 */
                unsigned getN() const;

                /**
                 * @brief This function returns a reference to the internal QFunction.
                 *
                 * @return The internal QFunction.
                 */
                const QFunction & getQFunction() const;

                /**
                 * @brief This function returns a reference to the referenced Model.
                 *
                 * @return The internal Model.
                 */
                const M & getModel() const;

            protected:
                unsigned N;
                const M & model_;
                QLearning qLearning_;

                // We use two structures because generally S*A is not THAT big, and we can definitely use
                // the O(1) insertion and O(1) sampling time.
                std::unordered_set<std::pair<size_t,size_t>, boost::hash<std::pair<size_t, size_t>>> visitedStatesActionsInserter_;
                std::vector<std::pair<size_t,size_t>> visitedStatesActionsSampler_;

                // Stuff for batch update
                mutable std::default_random_engine rand_;
        };

        template <typename M>
        DynaQ<M>::DynaQ(const M & m, double alpha, unsigned n) : N(n),
                                                                                  model_(m),
                                                                                  qLearning_(model_.getS(), model_.getA(), alpha, model_.getDiscount()),
                                                                                  rand_(Impl::Seeder::getSeed())
        {
            visitedStatesActionsInserter_.reserve(getS()*getA());
            visitedStatesActionsSampler_.reserve(getS()*getA());
        }

        template <typename M>
        void DynaQ<M>::stepUpdateQ(size_t s, size_t s1, size_t a, double rew) {
            qLearning_.stepUpdateQ(s, s1, a, rew);
            // O(1) insertion...
            auto result = visitedStatesActionsInserter_.insert(std::make_pair(s,a));
            if ( std::get<1>(result) )
                visitedStatesActionsSampler_.push_back(*std::get<0>(result));
        }

        template <typename M>
        void DynaQ<M>::batchUpdateQ() {
            if ( ! visitedStatesActionsSampler_.size() ) return;
            std::uniform_int_distribution<size_t> sampleDistribution_(0, visitedStatesActionsSampler_.size()-1);

            for ( unsigned i = 0; i < N; ++i ) {
                size_t s, s1, a;
                double rew;
                // O(1) sampling...
                std::tie(s,a) = visitedStatesActionsSampler_[sampleDistribution_(rand_)];

                std::tie(s1, rew) = model_.sample(s, a);
                qLearning_.stepUpdateQ(s, s1, a, rew);
            }
        }

        template <typename M>
        size_t DynaQ<M>::getS() const {
            return qLearning_.getS();
        }

        template <typename M>
        size_t DynaQ<M>::getA() const {
            return qLearning_.getA();
        }

        template <typename M>
        unsigned DynaQ<M>::getN() const {
            return N;
        }

        template <typename M>
        const QFunction & DynaQ<M>::getQFunction() const {
            return qLearning_.getQFunction();
        }
        template <typename M>
        const M & DynaQ<M>::getModel() const {
            return model_;
        }

        template <typename M>
        void DynaQ<M>::setLearningRate(double a) {
            qLearning_.setLearningRate(a);
        }

        template <typename M>
        double DynaQ<M>::getLearningRate() const {
            return qLearning_.getLearningRate();
        }
    }
}
#endif
