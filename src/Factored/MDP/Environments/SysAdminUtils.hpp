#ifndef AI_TOOLBOX_FACTORED_MDP_MULTI_AGENT_SYS_ADMIN_UTILS
#define AI_TOOLBOX_FACTORED_MDP_MULTI_AGENT_SYS_ADMIN_UTILS

#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

#include <AIToolbox/Utils/Core.hpp>

#include <algorithm>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This function builds the transition matrix for a single status state factor in the SysAdmin problem in case of action 0 (no-reboot).
     *
     * @param neighbors The number of neighbors of this agent.
     * @param id A number in [0, neighbors), indicating at which position in the tag this state-factor is.
     * @param pFailBase The base probability of failing.
     * @param pFailBonus The bonus probability of failing.
     * @param pDeadBase The base probability of dying.
     * @param pDeadBonus The bonus probability of dying.
     *
     * @return The transition matrix of size ((neighbors+1)^3, 3).
     */
    inline Matrix2D makeA0MatrixStatus(unsigned neighbors, size_t id, double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus) {
        using namespace SysAdminEnums;

        const unsigned neighborsCombinations = std::pow(3, neighbors + 1);
        Matrix2D retval(neighborsCombinations, 3);

        // We need the PartialFactorsEnumerator since the neighbors ids might be
        // lower and/or higher than this agent; so in order to iterate correctly to
        // fill the matrix we rely on the enumerator.
        PartialFactorsEnumerator e(Factors(neighbors + 1, 3));
        for (size_t i = 0; e.isValid(); e.advance(), ++i) {
            double bonus = 0.0;
            for (size_t n = 0; n < neighbors + 1; ++n) {
                if (n == id) continue;
                if (e->second[n] == Fail) bonus += pFailBonus;
                else if (e->second[n] == Dead) bonus += pDeadBonus;
            }
            bonus /= neighbors;

            const double pFail = pFailBase + bonus;
            const double pDead = pDeadBase + bonus;

            //                                     Good             Fail            Dead
            if (e->second[id] == Good)
                retval.row(i) <<         (1.0 - pFail),           pFail,            0.0;
            else if (e->second[id] == Fail)
                retval.row(i) <<                   0.0,   (1.0 - pDead),          pDead;
            else if (e->second[id] == Dead)
                retval.row(i) <<                   0.0,             0.0,            1.0;
        }

        return retval;
    }

    /**
     * @brief This function builds the transition matrix for a single status state factor in the SysAdmin problem in case of action 1 (reboot).
     *
     * Note that this does not depend on anything, since we are rebooting the
     * machine. Thus the matrix is always the same for all status state
     * factors.
     *
     * @return The transition matrix of size (3, 3).
     */
    inline Matrix2D makeA1MatrixStatus() {
        using namespace SysAdminEnums;

        Matrix2D retval(3, 3);

        //                  Good Fail Dead
        retval.row(Good) << 1.0, 0.0, 0.0;
        retval.row(Fail) << 1.0, 0.0, 0.0;
        retval.row(Dead) << 1.0, 0.0, 0.0;

        return retval;
    }

    /**
     * @brief This function builds the transition matrix for a single load state factor in the SysAdmin problem in case of action 1 (reboot).
     *
     * This function assumes that the status factor (on which it depends)
     * always comes before the load factor in the state space/tags.
     *
     * @param pLoad The probability of receiving a job.
     * @param pDoneG The probability of finishing a job when in good status.
     * @param pDoneF The probability of finishing a job when in failing status.
     *
     * @return The transition matrix of size (3*3, 3).
     */
    inline Matrix2D makeA0MatrixLoad(double pLoad, double pDoneG, double pDoneF) {
        using namespace SysAdminEnums;
        // States are Status + Idle, and we iterate over lower ids first, so the
        // matrix must be initialized by changing Status first.
        Matrix2D retval(3 * 3, 3);

        //                                      Idle             Load            Done
        retval.row(Idle * 3 + Good) << (1.0 - pLoad),           pLoad,            0.0;
        retval.row(Idle * 3 + Fail) << (1.0 - pLoad),           pLoad,            0.0;
        retval.row(Idle * 3 + Dead) <<           1.0,             0.0,            0.0;

        retval.row(Load * 3 + Good) <<           0.0,  (1.0 - pDoneG),         pDoneG;
        retval.row(Load * 3 + Fail) <<           0.0,  (1.0 - pDoneF),         pDoneF;
        retval.row(Load * 3 + Dead) <<           1.0,             0.0,            0.0;

        retval.row(Done * 3 + Good) <<           1.0,             0.0,            0.0;
        retval.row(Done * 3 + Fail) <<           1.0,             0.0,            0.0;
        retval.row(Done * 3 + Dead) <<           1.0,             0.0,            0.0;

        return retval;
    }

    /**
     * @brief This function builds the transition matrix for a single load state factor in the SysAdmin problem in case of action 1 (reboot).
     *
     * Note that this does not depend on anything, since we are rebooting the
     * machine. Thus the matrix is always the same for all load state
     * factors.
     *
     * @return The transition matrix of size (3, 3).
     */
    inline Matrix2D makeA1MatrixLoad() {
        using namespace SysAdminEnums;

        Matrix2D retval(3, 3);

        //                  Idle Load Done
        retval.row(Idle) << 1.0, 0.0, 0.0;
        retval.row(Load) << 1.0, 0.0, 0.0;
        retval.row(Done) << 1.0, 0.0, 0.0;

        return retval;
    }

    /**
     * @brief This function builds the reward function which is the same for all agents.
     *
     * The parameter can be built using the makeA0MatrixLoad(double, double, double) function.
     *
     * The reward matrix is all zero but for loaded states (since they are the
     * only ones from which it is possible to complete a job).
     *
     * We assume completing a job yelds 1.0 reward.
     *
     * @param la0Matrix The previously constructed transition matrix for the load factor.
     *
     * @return The reward matrix of size (3*3, 2).
     */
    inline Matrix2D makeRewardMatrix(const Matrix2D & la0Matrix) {
        using namespace SysAdminEnums;
        // All reward matrices for all agents are the same, so we build it here
        // once.
        //
        // In particular, we get 1 reward each time we get to a Done state.
        // However, our matrix of rewards is SxA (with no end states), so we need
        // to convert our definition of reward into SxA format.
        //
        // This means that we need to see which dependencies the Load state has:
        // both the previous Load and previous Status.
        Matrix2D rewardMatrix(3 * 3, 2);
        constexpr double finishReward = 1.0;
        rewardMatrix.setZero();

        // Basically, the only way we can get reward is by:
        // - Starting from the Load state (since it's the only one that can complete)
        // - Doing action 0;
        // - And ending up in the Done state.
        //
        // Remember that R(s,a) = sum_s1 T(s,a,s1) * R(s,a,s1)
        rewardMatrix(Load * 3 + Good, 0) = la0Matrix(Load * 3 + Good, Done) * finishReward;
        rewardMatrix(Load * 3 + Fail, 0) = la0Matrix(Load * 3 + Fail, Done) * finishReward;
        // This is zero as the transition from Load->Done is zero if the machine is dead.
        // rewardMatrix(Load * 3 + Dead, 0) = la0Matrix(Load * 3 + Dead, Done) * finishReward;

        return rewardMatrix;
    }

    /**
     * @brief This function returns a printable character of a machine's status.
     */
    inline char printMachineStatus(unsigned s) {
        using namespace SysAdminEnums;
        switch (s) {
            case Good: return 'g';
            case Fail: return 'f';
            default: return 'd';
        }
    }

    /**
     * @brief This function returns a printable character of a machine's load.
     */
    inline char printMachineLoad(unsigned l) {
        using namespace SysAdminEnums;
        switch (l) {
            case Idle: return 'i';
            case Load: return 'l';
            default: return 'd';
        }
    }

}

#endif
