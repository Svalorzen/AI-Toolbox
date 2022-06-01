#ifndef SUCCESSIVE_REJECTS_POLICY_HEADER_FILE
#define SUCCESSIVE_REJECTS_POLICY_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Bandit/Experience.hpp>
#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class implements the successive rejects algorithm.
     *
     * The successive rejects (SR) algorithm is a budget-based pure exploration
     * algorithm. Its goal is to simply recommend the best possible action
     * after its budget of pulls has been exhausted. The reward accumulated
     * during the exploration phase is irrelevant to the algorithm itself,
     * which is only focused on optimizing the quality of the final
     * recommendation.
     *
     * The way SR works is to split the available budget into phases. During
     * each phase, each arm is pulled a certain (nKNew_ - nKOld_) number of
     * times, which depends on the current phase. After these pulls, the arm
     * with the lowest empirical mean is removed from the pool of arms to be
     * evaluated.
     *
     * The algorithm keeps removing arms from the pool until a single arm
     * remains, which corresponds to the final recommended arm.
     */
    class SuccessiveRejectsPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param experience The experience gathering pull data of the bandit.
             * @param budget The overall pull budget for the exploration.
             */
            SuccessiveRejectsPolicy(const Experience & experience, unsigned budget);

            /**
             * @brief This function selects the current action to explore.
             *
             * Given how SR works, it simply recommends each arm
             * (nKNew_ - nKOld_) times, before cycling to the next action.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction() const override;

            /**
             * @brief This function updates the current phase, nK_, and prunes actions from the pool.
             *
             * This function must be called each timestep after the Experience has been updated.
             *
             * If needed, it will trigger pulling the next action in sequence.
             * If all actions have been pulled (nKNew_ - nKOld_) times, it will
             * increase the current phase, update both nK values and perform
             * the appropriate pruning using the current reward estimates
             * contained in the underlying Experience.
             */
            void stepUpdateQ();

            /**
             * @brief This function returns whether a single action remains in the pool.
             */
            bool canRecommendAction() const;

            /**
             * @brief If the pool has a single element, this function returns the best estimated action after the SR exploration process.
             */
            size_t recommendAction() const;

            /**
             * @brief This function returns the current phase.
             *
             * Note that if the exploration process is ended, the current phase will be equal to the number of actions.
             */
            size_t getCurrentPhase() const;

            /**
             * @brief This function returns the nK_ for the current phase.
             */
            size_t getCurrentNk() const;

            /**
             * @brief This function returns the nK_ for the previous phase.
             *
             * This is needed as the number of pulls for each arm in any given
             * phase is equal to the new Nk minus the old Nk.
             */
            size_t getPreviousNk() const;

            /**
             * @brief This function is fairly useless for SR, but it returns either 1.0 or 0.0 depending on which action is currently scheduled to be pulled.
             */
            virtual double getActionProbability(const size_t & a) const override;

            /**
             * @brief This function probably should not be called, but otherwise is what you would expect given the current timestep.
             */
            virtual Vector getPolicy() const override;

            /**
             * @brief This function returns a reference to the underlying Experience we use.
             *
             * @return The internal Experience reference.
             */
            const Experience & getExperience() const;

        private:
            void updateNks();

            const Experience & exp_;
            unsigned budget_;

            unsigned currentPhase_;
            size_t currentActionId_;
            unsigned currentArmPulls_;

            unsigned nKOld_, nKNew_;
            double logBarK_;
            std::vector<size_t> availableActions_;
    };
}

#endif

