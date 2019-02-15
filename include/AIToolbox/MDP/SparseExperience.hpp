#ifndef AI_TOOLBOX_MDP_SPARSE_EXPERIENCE_HEADER_FILE
#define AI_TOOLBOX_MDP_SPARSE_EXPERIENCE_HEADER_FILE

#include <iosfwd>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class keeps track of registered events and rewards.
     *
     * This class is a simple logger of events. It keeps track of both
     * the number of times a particular transition has happened, and the
     * total reward gained in any particular transition. However, it
     * does not record each event separately (i.e. you can't extract
     * the results of a particular transition in the past).
     *
     * The difference between this class and the MDP::Experience class is
     * that this class stores recorded events in sparse matrices. This
     * results in very high space savings when the state space of the
     * environment being logged is very high but only a small subset of
     * the states are really possible, at the cost of some efficiency
     * (possibly offset by cache savings).
     */
    class SparseExperience {
        public:
            using VisitTable = SparseTable3D;
            using VisitSumTable = SparseTable2D;
            using RewardMatrix = SparseMatrix3D;
            using RewardSumMatrix = SparseMatrix2D;

            /**
             * @brief Basic constructor.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             */
            SparseExperience(size_t s, size_t a);

            /**
             * @brief Compatibility setter.
             *
             * This function takes an arbitrary three dimensional
             * containers and tries to copy its contents into the
             * visits table.
             *
             * The container needs to support data access through
             * operator[]. In addition, the dimensions of the
             * container must match the ones specified during the
             * SparseExperience construction (for three dimensions: S,A,S).
             *
             * This is important, as this function DOES NOT perform
             * any size checks on the external containers.
             *
             * Note that if you are using a sparse container due
             * to the size of the state space, using this function
             * (which needs a dense container as an input) may
             * not be the optimal solution. If you have a sparse
             * container as a start you may look into converting
             * it into an Eigen Sparse container and feeding that
             * to this class.
             *
             * This function is provided so that it is easy to plug
             * this library into existing code-bases.
             *
             * @tparam V The external visits container type.
             * @param v The external visits container.
             */
            template <typename V>
            void setVisits(const V & v);

            /**
             * @brief Compatibility setter.
             *
             * This function takes an arbitrary three dimensional
             * container and tries to copy its contents into the
             * rewards matrix.
             *
             * The container needs to support data access through
             * operator[]. In addition, the dimensions of the
             * container must match the ones specified during the
             * SparseExperience construction (for three dimensions: S,A,S).
             *
             * This is important, as this function DOES NOT perform
             * any size checks on the external containers.
             *
             * Note that if you are using a sparse container due
             * to the size of the state space, using this function
             * (which needs a dense container as an input) may
             * not be the optimal solution. If you have a sparse
             * container as a start you may look into converting
             * it into an Eigen Sparse container and feeding that
             * to this class.
             *
             * This function is provided so that it is easy to plug
             * this library into existing code-bases.
             *
             * @tparam R The external rewards container type.
             * @param r The external rewards container.
             */
            template <typename R>
            void setRewards(const R & r);

            /**
             * @brief This function adds a new event to the recordings.
             *
             * @param s     Old state.
             * @param a     Performed action.
             * @param s1    New state.
             * @param rew   Obtained reward.
             */
            void record(size_t s, size_t a, size_t s1, double rew);

            /**
             * @brief This function resets all experienced rewards and transitions.
             */
            void reset();

            /**
             * @brief This function returns the current recorded visits for a transitions.
             *
             * @param s     Old state.
             * @param a     Performed action.
             * @param s1    New state.
             */
            unsigned long getVisits(size_t s, size_t a, size_t s1) const;

            /**
             * @brief This function returns the number of transitions recorded that start with the specified state and action.
             *
             * @param s     The initial state.
             * @param a     Performed action.
             *
             * @return The total number of transitions that start with the specified state-action pair.
             */
            unsigned long getVisitsSum(size_t s, size_t a) const;

            /**
             * @brief This function returns the cumulative rewards obtained from a specific transition.
             *
             * @param s     Old state.
             * @param a     Performed action.
             * @param s1    New state.
             */
            double getReward(size_t s, size_t a, size_t s1) const;

            /**
             * @brief This function returns the total reward obtained from transitions that start with the specified state and action.
             *
             * @param s     The initial state.
             * @param a     Performed action.
             *
             * @return The total number of transitions that start with the specified state-action pair.
             */
            double getRewardSum(size_t s, size_t a) const;

            /**
             * @brief This function returns the visits table for inspection.
             *
             * @return The visits table.
             */
            const VisitTable & getVisitTable() const;

            /**
             * @brief This function returns the rewards matrix for inspection.
             *
             * @return The rewards matrix.
             */
            const RewardMatrix & getRewardMatrix() const;

            /**
             * @brief This function returns the number of states of the world.
             *
             * @return The total number of states.
             */
            size_t getS() const;

            /**
             * @brief This function returns the number of available actions to the agent.
             *
             * @return The total number of actions.
             */
            size_t getA() const;

        private:
            size_t S, A;

            VisitTable visits_;
            VisitSumTable visitsSum_;

            RewardMatrix rewards_;
            RewardSumMatrix rewardsSum_;

            friend std::istream& operator>>(std::istream &is, SparseExperience &);
    };

    template <typename V>
    void SparseExperience::setVisits(const V & v) {
        for ( size_t a = 0; a < A; ++a )
            visits_[a].setZero();
        visitsSum_.setZero();

        for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a )
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            if ( v[s][a][s1] > 0 ) {
                visits_[a].insert(s, s1) = v[s][a][s1];
                visitsSum_.coeffRef(s,a) += v[s][a][s1];
            }
        }
        for ( size_t a = 0; a < A; ++a )
            visits_[a].makeCompressed();
        visitsSum_.makeCompressed();
    }

    template <typename R>
    void SparseExperience::setRewards(const R & r) {
        for ( size_t a = 0; a < A; ++a )
            rewards_[a].setZero();
        rewardsSum_.setZero();

        for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a )
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            if ( checkDifferentSmall(0.0, r[s][a][s1]) ) {
                rewards_[a].insert(s, s1) = r[s][a][s1];
                rewardsSum_.coeffRef(s,a) += r[s][a][s1];
            }
        }
        for ( size_t a = 0; a < A; ++a )
            rewards_[a].makeCompressed();
        rewardsSum_.makeCompressed();
    }
}

#endif
