#ifndef AI_TOOLBOX_MDP_DYNAQINTERFACE_HEADER_FILE
#define AI_TOOLBOX_MDP_DYNAQINTERFACE_HEADER_FILE

#include <AIToolbox/MDP/QLearning.hpp>
#include <AIToolbox/MDP/RLModel.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class represents an interface for algorithms in the form of DynaQ.
         */
        class DynaQInterface : public QLearning {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * This constructor requires the size of the MDP state and action
                 * spaces because they are most often needed in order to initialize
                 * and mantain the sampling queue.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param alpha The learning rate of the QLearning method.
                 * @param discount The discount of the QLearning method.
                 * @param n The number of sampling passes to do on the model upon batchUpdateQ().
                 */
                explicit DynaQInterface(size_t s, size_t a, double alpha = 0.5, double discount = 0.9, unsigned n = 50);

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
                 * @brief This function updates a QFunction based on simulated experience.
                 *
                 * @param m The RLModel we sample experience from.
                 * @param q The QFunction to update.
                 */
                virtual void batchUpdateQ(const RLModel & m, QFunction * q) = 0;
            protected:
                size_t S, A;
                unsigned N;
        };
    }
}
#endif
