#ifndef AI_TOOLBOX_BANDIT_ESRL_POLICY_HEADER_FILE
#define AI_TOOLBOX_BANDIT_ESRL_POLICY_HEADER_FILE

#include <AIToolbox/Bandit/Policies/LRPPolicy.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class models the Exploring Selfish Reinforcement Learning algorithm.
     *
     * This is a learning algorithm for common interest games. It tries to
     * consider both Nash equilibria and Pareto-optimal solution in order to
     * maximize the payoffs to the agents.
     *
     * The original algorithm can be modified in order to work with
     * non-cooperative games, but here we implement only the most general
     * version for cooperative games.
     *
     * An important point for this algorithm is that each agent only considers
     * its own payoffs, and in the cooperative case does not need to
     * communicate with the other agents.
     *
     * The idea is to repeatedly use the Linear Reward-Inaction algorithm to
     * converge and find a Nash equilibrium in the space of action, and then
     * cut that one from the action space and repeat the procedure. This would
     * recursively find out all Nash equilibra.
     *
     * This whole process is then repeated multiple times to ensure that most
     * of the equilibria have been explored.
     *
     * During each exploration step, a rolling average is maintained in order
     * to estimate the value of the action the LRI algorithm converged to.
     *
     * After all exploration phases have been done, the best action seen is
     * chosen and repeated forever during the final exploitation phase.
     */
    class ESRLPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param A The size of the action space.
             * @param a The learning parameter for Linear Reward-Inaction.
             * @param timesteps The number of timesteps per exploration phase.
             * @param explorationPhases The number of exploration phases before exploitation.
             * @param window The last number of timesteps to consider to obtain the learned action value during a single exploration phase.
             */
            ESRLPolicy(size_t A, double a, unsigned timesteps, unsigned explorationPhases, unsigned window);

            /**
             * @brief This function updates the ESRL policy based on the result of the action.
             *
             * Note that ESRL works with binary rewards: either the action
             * worked or it didn't.
             *
             * Environments where rewards are in R can be simulated: scale all
             * rewards to the [0,1] range, and stochastically obtain a success
             * with a probability equal to the reward. The result is equivalent
             * to the original reward function.
             *
             * This function both updates the internal LRI algorithm, and
             * checks whether a new exploration phase is warranted.
             *
             * @param a The action taken.
             * @param result Whether the action taken was a success, or not.
             */
            void stepUpdateP(size_t a, bool result);

            /**
             * @brief This function returns whether ESRL is now in the exploiting phase.
             *
             * This method returns whether ESRLPolicy has finished learning.
             * Once in the exploiting phase, the method won't learn anymore,
             * and will simply exploit the knowledge gained.
             *
             * Thus, if this method returns true, it won't be necessary anymore
             * to call the stepUpdateQ method (although it won't have any
             * effect to do so).
             *
             * @return Whether ESRLPolicy is in the exploiting phase.
             */
            bool isExploiting() const;

            /**
             * @brief This function chooses an action, following the policy distribution.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * @param a The selected action.
             *
             * @return The probability of taking the selected action.
             */
            virtual double getActionProbability(const size_t & a) const override;

            /**
             * @brief This function sets the a parameter.
             *
             * The a parameter determines the amount of learning on successful actions.
             *
             * @param a The new a parameter.
             */
            void setAParam(double a);

            /**
             * @brief This function will return the currently set a parameter.
             *
             * @return The currently set a parameter.
             */
            double getAParam() const;

            /**
             * @brief This function sets the required number of timesteps per exploration phase.
             *
             * @param t The new number of timesteps.
             */
            void setTimesteps(unsigned t);

            /**
             * @brief This function returns the currently set number of timesteps per exploration phase.
             *
             * @return The currently set number of timesteps.
             */
            unsigned getTimesteps() const;

            /**
             * @brief This function sets the required number of exploration phases before exploitation.
             *
             * @param p The new number of exploration phases.
             */
            void setExplorationPhases(unsigned p);

            /**
             * @brief This function returns the currently set number of exploration phases before exploitation.
             *
             * @return The currently set number of exploration phases.
             */
            unsigned getExplorationPhases() const;

            /**
             * @brief This function sets the size of the timestep window to compute the value of the action that ESRL is converging to.
             *
             * @param window The new size of the average window.
             */
            void setWindowSize(unsigned window);

            /**
             * @brief This function returns the currently set size of the timestep window to compute the value of an action.
             *
             * @return The currently set window size.
             */
            unsigned getWindowSize() const;

            /**
             * @brief This function returns a vector containing all probabilities of the policy.
             *
             * Ideally this function can be called only when there is a
             * repeated need to access the same policy values in an
             * efficient manner.
             */
            virtual Vector getPolicy() const override;

        private:
            // Whether we have learned enough to start exploiting.
            bool exploit_;
            size_t bestAction_;
            // Timesteps in current exploration phase in overall exploration phases.
            size_t timestep_, N_, explorations_, explorationPhases_;
            // Average value obtained in last window in the last exploration phase.
            double average_;
            size_t window_;

            // Values obtained for all actions.
            Vector values_;
            // Allowed actions in the current exploration phase.
            std::vector<size_t> allowedActions_;
            // Exploration learning policy to learn Nash equilibria.
            LRPPolicy lri_;
    };
}

#endif
