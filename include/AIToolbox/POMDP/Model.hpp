#ifndef AI_TOOLBOX_POMDP_MODEL_HEADER_FILE
#define AI_TOOLBOX_POMDP_MODEL_HEADER_FILE

#include <AIToolbox/Utils.hpp>
#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/POMDP/Types.hpp>

#include <random>
#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace POMDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<MDP::is_model<M>::value>::type>
        class Model;
#endif

        /**
         * @brief This class represents a Partially Observable Markov Decision Process.
         *
         * This class inherits from any valid MDP model type, so that it can
         * use its base methods, and it builds from those. Templated inheritance
         * was chosen to improve performance and keep code small, instead of
         * doing composition.
         *
         * @tparam M The particular MDP type that we want to extend.
         */
        template <typename M>
        class Model<M> : public M {
            public:
                using ObservationTable = Table3D;

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor initializes the Model so that all
                 * transitions happen with probability 0 but for transitions
                 * that bring back to the same state, no matter the action.
                 *
                 * All rewards are set to 0.
                 *
                 * All actions will return observation 0.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param o The number of possible observations the agent could make.
                 */
                Model(size_t s, size_t a, size_t o);

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor takes three arbitrary three dimensional
                 * containers and tries to copy their contents into the
                 * transitions, rewards and observations tables respectively.
                 *
                 * The containers need to support data access through
                 * operator[]. In addition, the dimensions of the
                 * containers must match the ones provided as arguments
                 * (for three dimensions: s,a,s/s,a,o).
                 *
                 * This is important, as this constructor DOES NOT perform
                 * any size checks on the external containers.
                 *
                 * Internal values of the containers will be converted to double,
                 * so these convertions must be possible.
                 *
                 * In addition, the transition and observation containers must contain
                 * valid transition functions
                 * \sa transitionCheck()
                 *
                 * \sa copyTable3D()
                 *
                 * @tparam T The external transition container type.
                 * @tparam R The external rewards container type.
                 * @tparam OF The external observations container type.
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param o The number of possible observations the agent could make.
                 * @param t The external transitions container.
                 * @param r The external rewards container.
                 * @param of The observation probability table.
                 */
                template <typename T, typename R, typename OF>
                Model(size_t s, size_t a, size_t o, const T & t, const R & r, const OF & of);

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor initializes the POMDP based on the values
                 * available from the underlying MDP and the provided
                 * information. The observation table dimensions have to match
                 * the specified state space, observation space and action space
                 * in this specific order.
                 *
                 * The table container will be traversed through operator[],
                 * with no bound checking. In addition, it has to specify a
                 * correct probability distribution.
                 *
                 * \sa transitionCheck()
                 * \sa copyTable3D()
                 *
                 * @tparam OF The external observation container type.
                 * @param underlyingMDP A reference to the underlying MDP for
                 * this POMDP. Make sure M supports copy construction!
                 * @param o The number of observations possible in the POMDP.
                 * @param of The observation probability table.
                 */
                template <typename OF>
                Model(const M & underlyingMDP, size_t o, const OF & of);

                /**
                 * @brief This function replaces the Model observation function with the one provided.
                 *
                 * The container needs to support data access through
                 * operator[]. In addition, the dimensions of the
                 * containers must match the ones provided as arguments
                 * (for three dimensions: s,a,o).
                 *
                 * This is important, as this constructor DOES NOT perform
                 * any size checks on the external containers.
                 *
                 * Internal values of the container will be converted to double,
                 * so that convertion must be possible.
                 *
                 * @tparam OF The external observations container type.
                 * @param table The external observations container.
                 */
                template <typename OF>
                void setObservationFunction(const OF & of);

                /**
                 * @brief This function returns the stored observation probability for the specified state-action pair.
                 *
                 * @param s1 The final state of the transition.
                 * @param a The action performed in the transition.
                 * @param o The recorded observation for the transition.
                 *
                 * @return The probability of the specified observation.
                 */
                double getObservationProbability(size_t s1, size_t a, size_t o) const;

                /**
                 * @brief This function *computes* the probability of obtaining an observation given an action and an initial belief.
                 *
                 * @param b The initial belief state.
                 * @param a The action performed.
                 * @param o The resulting observation.
                 *
                 * @return The probability of obtaining the specified observation.
                 */
                double getObservationProbability(const Belief & b, size_t o, size_t a) const;

                /**
                 * @brief Creates a new belief reflecting changes after an action and observation.
                 *
                 * This function creates a new belief since modifying a belief in place
                 * is not possible, as each cell update requires all values from the
                 * previous belief.
                 *
                 * @param b The old belief.
                 * @param a The action taken during the transition.
                 * @param o The observation registered.
                 */
                Belief updateBelief(const Belief & b, size_t a, size_t o) const;

                /**
                 * @brief This function returns the number of observations possible.
                 *
                 * @return The total number of observations.
                 */
                size_t getO() const;

                /**
                 * @brief This function returns the observation table for inspection.
                 *
                 * @return The rewards table.
                 */
                const ObservationTable & getObservationFunction() const;

            private:
                size_t O;
                ObservationTable observations_;
                // We need this because we don't know if our parent already has one,
                // and we wouldn't know how to access it!
                mutable std::default_random_engine rand_;
        };

        template <typename M>
        Model<M>::Model(size_t s, size_t a, size_t o) : M(s,a), O(o), observations_(boost::extents[this->getS()][this->getA()][O]) {
            for ( size_t s = 0; s < this->getS(); ++s )
                for ( size_t a = 0; a < this->getA(); ++a )
                    observations_[s][a][0] = 1.0;
        }

        template <typename M>
        template <typename T, typename R, typename OF>
        Model<M>::Model(size_t s, size_t a, size_t o, const T & t, const R & r, const OF & of) : M(s,a,t,r), O(o), observations_(boost::extents[this->getS()][this->getA()][O]),
                                                                               rand_(Impl::Seeder::getSeed())
        {
            setObservationFunction(of);
        }

        template <typename M>
        template <typename OF>
        Model<M>::Model(const M & underlyingMDP, size_t o, const OF & of) : M(underlyingMDP), O(o), observations_(boost::extents[this->getS()][this->getA()][O]),
                                                                               rand_(Impl::Seeder::getSeed())
        {
            setObservationFunction(of);
        }

        template <typename M>
        template <typename OF>
        void Model<M>::setObservationFunction(const OF & of) {
            for ( size_t s = 0; s < this->getS(); ++s )
                for ( size_t a = 0; a < this->getA(); ++a )
                    if ( ! isProbability(of[s][a], O) ) throw std::invalid_argument("Input observation table does not contain valid probabilities.");

            copyTable3D(of, observations_, this->getS(), this->getA(), O);
        }

        template <typename M>
        Belief Model<M>::updateBelief(const Belief & b, size_t a, size_t o) const {
            size_t S = this->getS();
            Belief br(S, 0.0);

            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                double sum = 0.0;
                for ( size_t s = 0; s < S; ++s )
                    sum += this->getTransitionProbability(s,a,s1) * b[s];

                br[s1] = getObservationProbability(s1,a,o) * sum;
            }

            return br;
        }

        template <typename M>
        double Model<M>::getObservationProbability(size_t s1, size_t a, size_t o) const {
            return observations_[s1][a][o];
        }

        template <typename M>
        size_t Model<M>::getO() const {
            return O;
        }

        template <typename M>
        const typename Model<M>::ObservationTable & Model<M>::getObservationFunction() const {
            return observations_;
        }
    }
}

#endif
