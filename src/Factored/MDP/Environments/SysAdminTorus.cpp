#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

#include "./SysAdminUtils.hpp"
#include <AIToolbox/Utils/Core.hpp>

#include <algorithm>

namespace AIToolbox::Factored::MDP {
    CooperativeModel makeSysAdminTorus(unsigned width, unsigned height,
        // Status transition params.
        double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus,
        // Load transition params.
        double pLoad, double pDoneG, double pDoneF);

    std::string printSysAdminTorus(const State & s);
}
