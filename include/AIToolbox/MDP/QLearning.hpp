#ifndef AI_TOOLBOX_MDP_QLEARNING_HEADER_FILE
#define AI_TOOLBOX_MDP_QLEARNING_HEADER_FILE

#include <stddef.h>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class represents the QLearning algorithm.
         */
        class QLearning {
            public:
                /**
                 * @brief Basic constructor.
                 * 
                 * The both the learning rate and the discount parameter 
                 * must be > 0.0 and <= 1.0, otherwise the constructor 
                 * will throw an std::runtime_error.
                 *
                 * @param alpha The learning rate of the QLearning method.
                 * @param discount The discount of the QLearning method.
                 */
                QLearning(double alpha = 0.5, double discount = 0.9);

                /**
                 * @brief Basic virtual destructor.
                 */
                virtual ~QLearning();

                /**
                 * @brief This function sets the learning rate parameter.
                 * 
                 * The learning rate parameter must be > 0.0 and <= 1.0,
                 * otherwise the function will do nothing.
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
                 * @brief This function sets the discount parameter.
                 * 
                 * The discount parameter must be > 0.0 and <= 1.0,
                 * otherwise the function will do nothing.
                 *
                 * @param d The new discount parameter.
                 */
                void setDiscount(double d);

                /**
                 * @brief This function will return the current set discount parameter.
                 *
                 * @return The currently set discount parameter.
                 */
                double getDiscount() const;

                /**
                 * @brief This function updates a given QFunction using the discount set during construction.
                 * 
                 * This function takes a single experience point and uses it to update
                 * a QFunction. This is a very efficient method to keep the QFunction
                 * up to date with the latest experience.
                 *
                 * @param s The previous state.
                 * @param s1 The new state.
                 * @param a The action performed.
                 * @param rew The reward obtained.
                 * @param q A pointer to the QFunction that is begin modified.
                 */
                virtual void stepUpdateQ(size_t s, size_t s1, size_t a, double rew, QFunction * q);
            protected:
                double alpha_;
                double discount_;
        };
    }
}
#endif
