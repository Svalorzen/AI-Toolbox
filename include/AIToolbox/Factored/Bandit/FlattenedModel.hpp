#ifndef AI_TOOLBOX_FACTORED_BANDIT_FLATTENED_MODEL_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_FLATTENED_MODEL_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Model.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class flattens a factored bandit model.
     *
     * This class allows to flatten a factored bandit model back into its
     * equivalent single-agent multi-armed bandit. This class is simply a
     * wrapper, and does not copy nor store the original model. Instead, all
     * conversions between joint-actions and flattened actions are done
     * on-the-fly as needed.
     *
     * Note that flattening the problem makes it harder, as the new bandit has
     * an effective action space equal to the full product of all the agents'
     * actions in the original problem, and does not get access to the
     * structure of the factorization.
     *
     * @tparam Dist The distribution to use for all arms.
     */
    template <typename Dist>
    class FlattenedModel {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param model The factored multi-armed bandit to wrap.
             */
            FlattenedModel(const Model<Dist> & model);

            /**
             * @brief This function samples the specified bandit arm.
             *
             * This function converts the input action into its equivalent
             * joint-action for the wrapped model. It then returns the sum of
             * the obtained reward vector.
             *
             * @param a The arm to sample.
             *
             * @return The sampled reward for the selected arm.
             */
            double sampleR(size_t a) const;

            /**
             * @brief This function converts the input action to its equivalent joint-action.
             *
             * @param a The input action.
             *
             * @return The equivalent joint-action for the wrapped bandit.
             */
            Action convertA(size_t a) const;

            /**
             * @brief This function returns the number of arms of the bandit.
             *
             * This value is pre-computed, not computed on the fly, to keep
             * this function fast.
             *
             * @return The number of arms of the bandit.
             */
            size_t getA() const;

            /**
             * @brief This function returns a reference to the wrapped factored bandit.
             *
             * @return The wrapped factored bandit.
             */
            const Model<Dist> & getModel() const;

        private:
            const Model<Dist> & model_;

            size_t A;
            mutable Action helper_;
    };

    template <typename Dist>
    FlattenedModel<Dist>::FlattenedModel(const Model<Dist> & model) :
            model_(model), A(factorSpace(model.getA())), helper_(model.getA().size())
    {}

    template <typename Dist>
    double FlattenedModel<Dist>::sampleR(size_t a) const {
        toFactors(model_.getA(), a, &helper_);
        return model_.sampleR(helper_).sum();
    }

    template <typename Dist>
    size_t FlattenedModel<Dist>::getA() const { return A; }
    template <typename Dist>
    const Model<Dist> & FlattenedModel<Dist>::getModel() const { return model_; }
}

#endif
