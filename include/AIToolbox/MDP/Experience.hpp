#ifndef AI_TOOLBOX_MDP_EXPERIENCE_HEADER_FILE
#define AI_TOOLBOX_MDP_EXPERIENCE_HEADER_FILE

#include <iosfwd>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/Utils/Core.hpp>

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
             * @brief Compatibility setter.
             *
             * This function takes an arbitrary three dimensional container and
             * tries to copy its contents into the visits table.
             *
             * The container needs to support data access through operator[].
             * In addition, the dimensions of the container must match the ones
             * specified during the Experience construction (for three
             * dimensions: S,A,S).
             *
             * This is important, as this function DOES NOT perform any size
             * checks on the external containers.
             *
             * This function is provided so that it is easy to plug this
             * library into existing code-bases.
             *
             * @tparam V The external visits container type.
             * @param v The external visits container.
             */
            template <typename V>
            void setVisitsTable(const V & v);

            /**
             * @brief Compatibility setter.
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
             * This function is provided so that it is easy to plug this
             * library into existing code-bases.
             *
             * @tparam R The external rewards container type.
             * @param r The external rewards container.
             */
            template <typename R>
            void setRewardMatrix(const R & r);

            /**
             * @brief Compatibility setter.
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
             * This function is provided so that it is easy to plug this
             * library into existing code-bases.
             *
             * @tparam MM The external M2 container type.
             * @param m The external M2 container.
             */
            template <typename MM>
            void setM2Matrix(const MM & mm);

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

    template <typename V>
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

    template <typename R>
    void Experience::setRewardMatrix(const R & r) {
        for ( size_t s = 0; s < S; ++s )
            for ( size_t a = 0; a < A; ++a )
                rewards_(s, a) = r[s][a];
    }

    template <typename MM>
    void Experience::setM2Matrix(const MM & m) {
        for ( size_t s = 0; s < S; ++s )
            for ( size_t a = 0; a < A; ++a )
                M2s_(s, a) = m[s][a];
    }
}

#endif
