#ifndef AI_TOOLBOX_MDP_RLMODEL_HEADER_FILE
#define AI_TOOLBOX_MDP_RLMODEL_HEADER_FILE

#include <tuple>
#include <random>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Experience.hpp>
#include <AIToolbox/MDP/Model.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class models Experience as a Markov Decision Process.
         *
         * This class normalizes an Experience object to produce a transition function
         * and a reward function. The transition function is guaranteed to be a correct
         * probability function, as in the sum of the probabilities of all transitions
         * from a particular state and a particular action is always 1.
         * Each instance is not directly synced with the supplied Experience object.
         * This is to avoid possible overheads, as the user can optimize better
         * depending on their use case. See sync().
         */
        class RLModel : public Model {
            public:
                /**
                 * @brief Constructor using previous Experience.
                 *
                 * This constructor simply selects the Experience that will
                 * be used to learn an MDP Model from the data.
                 *
                 * After the constructor the user needs to manually sync() 
                 * the RLModel to the Experience in order for the transitions
                 * and rewards tables to be correctly computed. 
                 * 
                 * The default transition function defines a transition of 
                 * probability 1 for each transition to state to itself,
                 * using action 0.
                 *
                 * The default reward function is 0.
                 *
                 * @param exp The base Experience of the model.
                 */
                RLModel(const Experience & exp);

                /**
                 * @brief Constructor that sets initial transition and reward tables.
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
                 * the constraint described in the Model::mdpCheck() function.
                 *
                 * @tparam T The external transition container type.
                 * @tparam R The external rewards container type.
                 * @param exp The base Experience of the model.
                 * @param t The external transitions container. 
                 * @param r The external rewards container. 
                 */
                template <typename T, typename R>
                RLModel(const Experience & exp, T t, R r);

                /**
                 * @brief This function syncs the whole RLModel to the underlying Experience.
                 *
                 * Since use cases in AI are very varied, one may not want to update
                 * its RLModel for each single transition experienced by the agent. To
                 * avoid this we leave to the user the task of syncing between the
                 * underlying Experience and the RLModel, as he/she sees fit.
                 *
                 * After this function is run the transition and reward functions
                 * will accurately reflect the state of the underlying Experience.
                 */
                void sync();

                /**
                 * @brief This function syncs a state action pair in the RLModel to the underlying Experience.
                 *
                 * Since use cases in AI are very varied, one may not want to update
                 * its RLModel for each single transition experienced by the agent. To
                 * avoid this we leave to the user the task of syncing between the
                 * underlying Experience and the RLModel, as he/she sees fit.
                 *
                 * This function updates a single state action pair with the underlying
                 * Experience. This function is offered to avoid having to recompute the
                 * whole RLModel if the user knows that only few transitions have been
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
                 * @brief This function enables inspection of the underlying Experience of the RLModel.
                 *
                 * @return The underlying Experience of the RLModel.
                 */
                const Experience & getExperience() const;

            protected:
                using Model::S;
                using Model::A;

                const Experience & experience_;
        };

        template <typename T, typename R>
        RLModel::RLModel(const Experience & exp, T t, R r) : Model(exp.getS(), exp.getA()), experience_(exp) {
            copyTable3D(t, transitions_, S, S, A);
            copyTable3D(r, rewards_, S, S, A);
        }
    }
}

#endif
