#ifndef AI_TOOLBOX_FACTORED_MDP_MULTI_AGENT_SYS_ADMIN_UTILS
#define AI_TOOLBOX_FACTORED_MDP_MULTI_AGENT_SYS_ADMIN_UTILS

#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

#include <AIToolbox/Utils/Core.hpp>

#include <algorithm>

namespace AIToolbox::Factored::MDP {
    // The status evolution of a machine only depends on its own status, plus the
    // status of its neighbors
    Matrix2D makeA0MatrixStatus(unsigned neighbors, size_t id, double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus) {
        using namespace SysAdminEnums;

        const unsigned neighborsCombinations = std::pow(3, neighbors);
        Matrix2D retval(3 * neighborsCombinations, 3);

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

    Matrix2D makeA1MatrixStatus() {
        using namespace SysAdminEnums;

        Matrix2D retval(3, 3);

        //                  Good Fail Dead
        retval.row(Good) << 1.0, 0.0, 0.0;
        retval.row(Fail) << 1.0, 0.0, 0.0;
        retval.row(Dead) << 1.0, 0.0, 0.0;

        return retval;
    }

    // The load of a machine only depends on its own status and its own load.
    Matrix2D makeA0MatrixLoad(double pLoad, double pDoneG, double pDoneF) {
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

    Matrix2D makeA1MatrixLoad() {
        using namespace SysAdminEnums;

        Matrix2D retval(3, 3);

        //                  Idle Load Done
        retval.row(Idle) << 1.0, 0.0, 0.0;
        retval.row(Load) << 1.0, 0.0, 0.0;
        retval.row(Done) << 1.0, 0.0, 0.0;

        return retval;
    }
}

#endif
