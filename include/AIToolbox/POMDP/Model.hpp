#ifndef AI_TOOLBOX_POMDP_MODEL_HEADER_FILE
#define AI_TOOLBOX_POMDP_MODEL_HEADER_FILE

#include <AIToolbox/Utils.hpp>
#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<MDP::is_model<M>::value>::type>
        class Model;
#endif

        template <typename M>
        class Model<M> {
            public:
                using ObservationTable = Table3D;

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
                 * @tparam O The external observation container type.
                 * @param underlyingMDP A reference to the underlying MDP for
                 * this POMDP.
                 * @param o The number of observations possible in the POMDP.
                 * @param table The observation probability table.
                 */
                template <typename O>
                Model(const M & underlyingMDP, size_t o, const O & table);

                /**
                 * @brief This function returns the stored observation probability for the specified state-action pair.
                 *
                 * @param s The initial state of the transition.
                 * @param o The recorded observation for the transition.
                 * @param a The action performed in the transition.
                 *
                 * @return The probability of the specified observation.
                 */
                double getObservationProbability(size_t s, size_t o, size_t a) const;
                double getObservationProbability(const Belief & b, size_t o, size_t a) const;

                Belief updateBelief(const Belief & b, size_t a, size_t o) const;

                double getTransitionProbability(const Belief & b, size_t a) const;
                double getExpectedReward(const Belief & b, size_t a, size_t o) const;

                const M & getMDP() const;
            private:
                const M & mdp_;

                size_t O;
                ObservationTable observations_;
        };
    }
}

#endif
