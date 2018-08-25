#ifndef AI_TOOLBOX_MDP_DYNAQ_HEADER_FILE
#define AI_TOOLBOX_MDP_DYNAQ_HEADER_FILE

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Algorithms/QLearning.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

#include <boost/functional/hash.hpp>
#include <utility>
#include <unordered_set>
#include <vector>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the DynaQ algorithm.
     *
     * This algorithm is a simple extension to the QLearning algorithm.
     * What it does is it keeps track of every experienced state-action
     * pair. Each QFunction update is exactly equivalent to the QLearning
     * one, however this algorithm allows for an additional learning phase
     * that can take place, time permitting, before the agent takes another
     * action.
     *
     * The state-action pairs we already explored are thus known as
     * possible, and so we use the generative model to obtain more and more
     * data about them. This, of course, requires that the model be sampled
     * from, in constrast with QLearning which does not require this.
     *
     * The algorithm selects randomly which state action pairs to try again
     * from.
     */
    template <typename M>
    class DynaQ {
        static_assert(is_generative_model_v<M>, "This class only works for generative MDP models!");

        public:
            /**
             * @brief Basic constructor.
             *
             * @param m The model to be used to update the QFunction.
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
             * @param a The action performed.
             * @param s1 The new state.
             * @param rew The reward obtained.
             */
            void stepUpdateQ(size_t s, size_t a, size_t s1, double rew);

            /**
             * @brief This function updates a QFunction based on simulated experience.
             *
             * In DynaQ we sample N times from already experienced
             * state-action pairs, and we update the resulting QFunction as
             * if this experience was actually real.
             *
             * The idea is that since we know which state action pairs we already
             * explored, we know that whose pairs are actually possible. Thus we
             * use the generative model to sample them again, and obtain a better
             * estimate of the QFunction.
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

        private:
            unsigned N;
            const M & model_;
            QLearning qLearning_;

            // We use two structures because generally S*A is not THAT big, and we can definitely use
            // the O(1) insertion and O(1) sampling time.
            std::unordered_set<std::pair<size_t,size_t>, boost::hash<std::pair<size_t, size_t>>> visitedStatesActionsInserter_;
            std::vector<std::pair<size_t,size_t>> visitedStatesActionsSampler_;

            // Stuff for batch update
            mutable RandomEngine rand_;
    };

    template <typename M>
    DynaQ<M>::DynaQ(const M & m, const double alpha, const unsigned n) :
            N(n), model_(m), qLearning_(model_, alpha), rand_(Impl::Seeder::getSeed())
    {
        visitedStatesActionsInserter_.reserve(model_.getS()*model_.getA());
        visitedStatesActionsSampler_.reserve(model_.getS()*model_.getA());
    }

    template <typename M>
    void DynaQ<M>::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const double rew) {
        qLearning_.stepUpdateQ(s, a, s1, rew);
        // O(1) insertion...
        const auto result = visitedStatesActionsInserter_.insert(std::make_pair(s,a));
        if ( std::get<1>(result) )
            visitedStatesActionsSampler_.push_back(*std::get<0>(result));
    }

    template <typename M>
    void DynaQ<M>::batchUpdateQ() {
        if ( ! visitedStatesActionsSampler_.size() ) return;
        std::uniform_int_distribution<size_t> sampleDistribution_(0, visitedStatesActionsSampler_.size()-1);

        for ( unsigned i = 0; i < N; ++i ) {
            // O(1) sampling...
            const auto [s,a] = visitedStatesActionsSampler_[sampleDistribution_(rand_)];
            const auto [s1, rew] = model_.sample(s, a);

            qLearning_.stepUpdateQ(s, a, s1, rew);
        }
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
    void DynaQ<M>::setLearningRate(const double a) {
        qLearning_.setLearningRate(a);
    }

    template <typename M>
    double DynaQ<M>::getLearningRate() const {
        return qLearning_.getLearningRate();
    }
}

#endif
