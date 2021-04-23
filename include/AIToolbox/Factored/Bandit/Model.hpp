#ifndef AI_TOOLBOX_FACTORED_BANDIT_MODEL_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_MODEL_HEADER_FILE

#include <AIToolbox/Bandit/Model.hpp>
#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class represents a factored multi-armed bandit.
     *
     * A factored multi-armed bandit is a specific bandit class, where the
     * reward function is factored into independent components, each of which
     * only depends on a subset of agents. The goal is generally to maximize
     * the sum of the rewards of all local arms.
     *
     * It effectively behaves as a collection of multi-armed bandits, aside
     * from the fact that the action each agent takes will be the same in all
     * bandits that it participates into. Each "local" bandit's effective
     * action will be the combination of all participating agents.
     *
     * This structure can make learning learning how to act much more
     * efficient, as exploiting the factorization allows to extract more
     * information from each joint action performed by the agents.
     *
     * @tparam Dist The distribution to use for all local arms.
     */
    template <typename Dist>
    class Model {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor creates the factored multi-armed bandit from a
             * set of standard bandits, each associated with a group of agents.
             *
             * Note that the action space of each bandit must be equal to the
             * product of the action spaces of all agents in its group. For
             * example, a bandit associated with agents with action spaces 2,
             * 3, 2 should have 12 arms in total.
             *
             * @param A The joint action space.
             * @param deps The agents associated with each bandit.
             * @param arms The local bandits to use.
             */
            template <typename... TupleArgs>
            Model(Action A, std::vector<PartialKeys> deps, std::vector<AIToolbox::Bandit::Model<Dist>> arms);

            /**
             * @brief This function samples the specified joint bandit arm.
             *
             * @param a The joint arm to sample.
             *
             * @return A vector containing the rewards of each local arm.
             */
            Rewards sampleR(const Action & a) const;

            /**
             * @brief This function returns the joint action space.
             */
            const Action & getA() const;

            /**
             * @brief This function returns a reference to the agent groupings.
             */
            const std::vector<PartialKeys> & getGroups() const;

            /**
             * @brief This function returns a reference to the internal local arms.
             */
            const std::vector<AIToolbox::Bandit::Model<Dist>> & getArms() const;

        private:
            Action A;
            std::vector<PartialKeys> groups_;

            mutable std::vector<AIToolbox::Bandit::Model<Dist>> arms_;
    };

    template <typename Dist>
    template <typename... TupleArgs>
    Model<Dist>::Model(Action a, std::vector<PartialKeys> deps, std::vector<AIToolbox::Bandit::Model<Dist>> arms) :
            A(std::move(a)), groups_(std::move(deps)), arms_(std::move(arms))
    {
        // Sanity checks
        //
        // - The number of groups is equal to the number of local arms.
        // - Each local arm has an action space equal to the product
        //   of its participating agents.
        assert(groups_.size() == arms_.size());

        for (size_t i = 0; i < groups_.size(); ++i) {
            const auto bSize = factorSpacePartial(groups_[i], A);
            (void)bSize;

            assert(bSize == arms_[i].getA());
        }
    }

    template <typename Dist>
    Rewards Model<Dist>::sampleR(const Action & a) const {
        Rewards rews(groups_.size());

        for (size_t i = 0; i < groups_.size(); ++i) {
            const auto aid = toIndexPartial(groups_[i], A, a);

            rews[i] = arms_[i].sampleR(aid);
        }

        return rews;
    }

    template <typename Dist>
    const Action & Model<Dist>::getA() const { return A; }
    template <typename Dist>
    const std::vector<PartialKeys> & Model<Dist>::getGroups() const { return groups_; }
    template <typename Dist>
    const std::vector<AIToolbox::Bandit::Model<Dist>> & Model<Dist>::getArms() const { return arms_; }
}

#endif
