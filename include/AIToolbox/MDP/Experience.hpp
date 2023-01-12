#ifndef AI_TOOLBOX_MDP_EXPERIENCE_HEADER_FILE
#define AI_TOOLBOX_MDP_EXPERIENCE_HEADER_FILE

#include <iosfwd>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/TypeTraits.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class keeps track of registered events and rewards.
     *
     * This class is a simple aggregator of events. It keeps track of both the
     * number of times a particular transition has been visited, and the
     * average reward gained per state-action pair (i.e. the maximum likelihood
     * estimator of a QFunction from the data). It also computes the M2
     * statistic for the rewards (avg sum of squares minus square avg).
     *
     * It does not record each event separately (i.e. you can't extract the
     * results of a particular transition in the past).
     */
    class Experience {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param S The number of states of the world.
             * @param A The number of actions available to the agent.
             */
            Experience(size_t S, size_t A);

            /**
             * @brief This function sets the internal visits table to the input.
             *
             * This function takes an arbitrary three dimensional container and
             * tries to copy its contents into the visits table. It
             * automatically updates the visitsSum table as well.
             *
             * The container needs to support data access through operator[].
             * In addition, the dimensions of the container must match the ones
             * specified during the Experience construction (for three
             * dimensions: S,A,S).
             *
             * This is important, as this function DOES NOT perform any size
             * checks on the external containers.
             *
             * @tparam V The external visits container type.
             * @param v The external visits container.
             */
            template <IsNaive3DTable V>
            void setVisitsTable(const V & v);

            /**
             * @brief This function sets the internal visits table to the input.
             *
             * This function copies the input Table3D into the visits table.
             * It automatically updates the visitsSum table as well.
             *
             * The dimensions of the input must match the ones
             * specified during the Experience construction (for three
             * dimensions: A, S, S).
             * BE CAREFUL. The tables MUST be SxS, while the std::vector
             * containing them MUST be of size A.
             *
             * This is important, as this function DOES NOT perform any size
             * checks on the external containers.
             *
             * @param v The external visits container.
             */
            void setVisitsTable(const Table3D & v);

            /**
             * @brief This function sets the internal reward matrix to the input.
             *
             * This function takes an arbitrary two dimensional container and
             * tries to copy its contents into the rewards matrix.
             *
             * The container needs to support data access through operator[].
             * In addition, the dimensions of the container must match the ones
             * specified during the Experience construction (for two
             * dimensions: S,A).
             *
             * This is important, as this function DOES NOT perform any size
             * checks on the external containers.
             *
             * @tparam R The external rewards container type.
             * @param r The external rewards container.
             */
            template <IsNaive2DMatrix R>
            void setRewardMatrix(const R & r);

            /**
             * @brief This function sets the internal reward matrix to the input.
             *
             * The dimensions of the input must match the ones
             * specified during the Experience construction (for two
             * dimensions: S, A).
             * BE CAREFUL. The tables MUST be SxS, while the std::vector
             * containing them MUST be of size A.
             *
             * This is important, as this function DOES NOT perform any size
             * checks on the external containers.
             *
             * @param r The external rewards container.
             */
            void setRewardMatrix(const Matrix2D & r);

            /**
             * @brief This function sets the internal m2 matrix to the input.
             *
             * This function takes an arbitrary two dimensional container and
             * tries to copy its contents into the M2 matrix.
             *
             * The container needs to support data access through operator[].
             * In addition, the dimensions of the container must match the ones
             * specified during the Experience construction (for two
             * dimensions: S,A).
             *
             * This is important, as this function DOES NOT perform any size
             * checks on the external containers.
             *
             * @tparam MM The external M2 container type.
             * @param mm The external M2 container.
             */
            template <IsNaive2DMatrix MM>
            void setM2Matrix(const MM & mm);

            /**
             * @brief This function sets the internal m2 matrix to the input.
             *
             * The dimensions of the input must match the ones
             * specified during the Experience construction (for two
             * dimensions: S, A).
             * BE CAREFUL. The tables MUST be SxS, while the std::vector
             * containing them MUST be of size A.
             *
             * This is important, as this function DOES NOT perform any size
             * checks on the external containers.
             *
             * @param mm The external M2 container.
             */
            void setM2Matrix(const Matrix2D & mm);

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
             * @brief This function resets all experienced rewards, transitions and M2s.
             */
            void reset();

            /**
             * @brief This function returns the number of times the record function has been called.
             *
             * @return The number of recorded timesteps.
             */
            unsigned long getTimesteps() const;

            /**
             * @brief This function returns the current recorded visits for a transition.
             *
             * @param s Old state.
             * @param a Performed action.
             * @param s1 New state.
             */
            unsigned long getVisits(size_t s, size_t a, size_t s1) const;

            /**
             * @brief This function returns the current recorded visits for a state-action pair.
             *
             * @param s Old state.
             * @param a Performed action.
             */
            unsigned long getVisitsSum(size_t s, size_t a) const;

            /**
             * @brief This function returns the average reward for a state-action pair.
             *
             * @param s Old state.
             * @param a Performed action.
             */
            double getReward(size_t s, size_t a) const;

            /**
             * @brief This function returns the M2 statistic for a state-action pair.
             *
             * @param s Old state.
             * @param a Performed action.
             */
            double getM2(size_t s, size_t a) const;

            /**
             * @brief This function returns the visits table for inspection.
             *
             * @return The visits table.
             */
            const Table3D & getVisitsTable() const;

            /**
             * @brief This function returns the visits table for inspection.
             *
             * @param a The action requested.
             *
             * @return The visits table.
             */
            const Table2D & getVisitsTable(size_t a) const;

            /**
             * @brief This function returns the visits sum table for inspection.
             *
             * This table contains per state-action pair visit counts.
             *
             * @return The visits sum table.
             */
            const Table2D & getVisitsSumTable() const;

            /**
             * @brief This function returns the rewards matrix for inspection.
             *
             * The reward matrix contains the current average rewards computed for each state-action pairs.
             *
             * @return The rewards matrix.
             */
            const QFunction & getRewardMatrix() const;

            /**
             * @brief This function returns the rewards squared matrix for inspection.
             *
             * @return The rewards squared matrix.
             */
            const Matrix2D & getM2Matrix() const;

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

            Table3D visits_;
            Table2D visitsSum_;
            Matrix2D rewards_;
            Matrix2D M2s_;
            unsigned long timesteps_;

            friend std::istream& operator>>(std::istream &is, Experience &);
    };

    template <IsNaive3DTable V>
    void Experience::setVisitsTable(const V & v) {
        visitsSum_.setZero();
        for ( size_t s = 0; s < S; ++s ) {
            for ( size_t a = 0; a < A; ++a ) {
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    visits_[a](s, s1) = v[s][a][s1];
                    visitsSum_(s, a) += v[s][a][s1];
                }
            }
        }
    }

    template <IsNaive2DMatrix R>
    void Experience::setRewardMatrix(const R & r) {
        for ( size_t s = 0; s < S; ++s )
            for ( size_t a = 0; a < A; ++a )
                rewards_(s, a) = r[s][a];
    }

    template <IsNaive2DMatrix MM>
    void Experience::setM2Matrix(const MM & m) {
        for ( size_t s = 0; s < S; ++s )
            for ( size_t a = 0; a < A; ++a )
                M2s_(s, a) = m[s][a];
    }
}

#endif
