#ifndef AI_TOOLBOX_RLMDP_HEADER_FILE
#define AI_TOOLBOX_RLMDP_HEADER_FILE

#include <tuple>
#include <random>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Experience.hpp>
#include <AIToolbox/MDP.hpp>

namespace AIToolbox {
    /**
     * @brief This class models Experience as a Markov Decision Process.
     *
     * This class normalizes an Experience object to produce a transition function
     * and a reward function. The transition function is guaranteed to be a correct
     * probability function, as in the sum of the probabilities of all transitions
     * from a particular state and a particular action is always 1.
     * Each instance is not directly synced with the supplied Experience object.
     * This is to avoid possible overheads, as the user can optimize better
     * depending on their use case. See update().
     */
    class RLMDP : public MDP {
        public:
            /**
             * @brief Constructor using previous Experience.
             *
             * This constructor simply selects the Experience that will
             * be used to learn an MDP model from the data.
             *
             * After the constructor the user needs to manually sync() 
             * the RLMDP to the Experience in order for the transitions
             * and rewards tables to be correctly computed. 
             *
             * @param exp The base Experience of the model.
             */
            RLMDP(const Experience & exp);

            /**
             * @brief Constructor that sets default transition and reward tables.
             *
             * This constructor takes two arbitrary three dimensional
             * containers and tries to copy their contents into the
             * transitions and rewards tables respectively. 
             *
             * The containers need to support data access through 
             * operator[]. In addition, the dimensions of the
             * containers must match the ones of the provided
             * Experience (for three dimensions: s,s,a).
             * 
             * This is important, as this constructor DOES NOT perform
             * any size checks on the external containers.
             * 
             * In addition, the transition container must respect
             * the constraint described in the MDP::mdpCheck() function.
             *
             * @tparam T The external transition container type.
             * @tparam R The external rewards container type.
             * @param exp The base Experience of the model.
             * @param t The external transitions container. 
             * @param r The external rewards container. 
             */
            template <typename T, typename R>
            RLMDP(const Experience & exp, T t, R r);

            /**
             * @brief This function syncs the RLMDP to the underlying Experience.
             *
             * Since use cases in AI are very varied, one may not want to update
             * its RLMDP for each single transition experienced by the agent. To
             * avoid this we leave to the user the task of syncing between the
             * underlying Experience and the RLMDP, as he/she sees fit.
             *
             * After this function is run the transition and reward functions
             * will accurately reflect the state of the underlying Experience.
             */
            void sync();

            /**
             * @brief This function syncs a state action pair in the RLMDP to the underlying Experience.
             *
             * Since use cases in AI are very varied, one may not want to update
             * its RLMDP for each single transition experienced by the agent. To
             * avoid this we leave to the user the task of syncing between the
             * underlying Experience and the RLMDP, as he/she sees fit.
             *
             * This function updates a single state action pair with the underlying
             * Experience. This function is offered to avoid having to recompute the
             * whole RLMDP if the user knows that only few transitions have been
             * experienced by the agent.
             *
             * After this function is run the transition and reward functions
             * will accurately reflect the state of the underlying Experience
             * for the specified state action pair.
             *
             * @param s The state that needs to be synced.
             * @param a The action that needs to be synced.
             */
            void sync(size_t s, size_t a);

            /**
             * @brief This function enables inspection of the underlying Experience of the RLMDP.
             *
             * @return The underlying Experience of the RLMDP.
             */
            const Experience & getExperience() const;

        protected:
            const Experience & experience_;
    };
}

#endif
