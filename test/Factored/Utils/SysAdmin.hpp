#ifndef AI_TOOLBOX_FACTORED_MDP_MULTI_AGENT_SYS_ADMIN
#define AI_TOOLBOX_FACTORED_MDP_MULTI_AGENT_SYS_ADMIN

#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>

#include <algorithm>

namespace ai = AIToolbox;
namespace aif = AIToolbox::Factored;
namespace afm = AIToolbox::Factored::MDP;

enum MachineStatus {
    Good = 0,
    Fail,
    Dead
};

enum MachineLoad {
    Idle = 0,
    Load,
    Done
};

// The status evolution of a machine only depends on its own status, plus the
// status of its neighbors
ai::Matrix2D makeA0MatrixStatus(unsigned neighbors, size_t id, double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus) {
    const unsigned neighborsCombinations = std::pow(3, neighbors);
    ai::Matrix2D retval(3 * neighborsCombinations, 3);

    // We need the PartialFactorsEnumerator since the neighbors ids might be
    // lower and/or higher than this agent; so in order to iterate correctly to
    // fill the matrix we rely on the enumerator.
    aif::PartialFactorsEnumerator e(aif::Factors(neighbors + 1, 3));

    size_t i = 0;
    while (e.isValid()) {
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

        ++i;
        e.advance();
    }

    return retval;
}

ai::Matrix2D makeA1MatrixStatus() {
    ai::Matrix2D retval(3, 3);

    //                  Good Fail Dead
    retval.row(Good) << 1.0, 0.0, 0.0;
    retval.row(Fail) << 1.0, 0.0, 0.0;
    retval.row(Dead) << 1.0, 0.0, 0.0;

    return retval;
}

// The load of a machine only depends on its own status and its own load.
ai::Matrix2D makeA0MatrixLoad(double pLoad, double pDoneG, double pDoneF) {
    // States are Status + Idle, and we iterate over lower ids first, so the
    // matrix must be initialized by changing Status first.
    ai::Matrix2D retval(3 * 3, 3);

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

ai::Matrix2D makeA1MatrixLoad() {
    ai::Matrix2D retval(3, 3);

    //                  Idle Load Done
    retval.row(Idle) << 1.0, 0.0, 0.0;
    retval.row(Load) << 1.0, 0.0, 0.0;
    retval.row(Done) << 1.0, 0.0, 0.0;

    return retval;
}

afm::CooperativeModel makeSysAdminBiRing(unsigned agents,
    // Status transition params.
    double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus,
    // Load transition params.
    double pLoad, double pDoneG, double pDoneF)
{
    // Parameters for this network type:
    // In a ring we have 2 neighbors.
    constexpr unsigned neighbors = 2;

    // We factor the state space into two variables per each agent: status and
    // load. Each of them can assume 3 different values.
    aif::State S(agents * 2);
    std::fill(std::begin(S), std::end(S), 3);

    // Each agent has a single action, so the size of the action space is equal
    // to the number of agents.
    aif::Action A(agents);
    std::fill(std::begin(A), std::end(A), 2);

    // All matrices but the a0 status transitions do not depend on the
    // neighbors, so we can create them only once and just copy them when we
    // need them.
    const auto sa1Matrix = makeA1MatrixStatus();
    const auto la0Matrix = makeA0MatrixLoad(pLoad, pDoneG, pDoneF);
    const auto la1Matrix = makeA1MatrixStatus();

    auto ddn = aif::FactoredDDN();
    for (size_t a = 0; a < agents; ++a) {
        // Here, for each action, we have to create two transition nodes: one
        // for the status of the machine, and another for the load.
        // Both nodes only depend on the action of its agent.

        // Status node, only depends on the action of 'a'
        aif::FactoredDDN::Node nodeStatus{{a}, {}};

        // Status nodes for action 0 (do nothing) and action 1 (restart) respectively.
        // Node that the transition node for action 0 depends on the neighbors,
        // since whether they are failing or not affects whether this machine
        // will fail or not. If we reset, we don't really care.
        aif::DBN::Node sa0{{}, {}};
        unsigned neighborId;
        // Set the correct dependencies for the ring
        if (a == 0) {
            sa0.tag = {0, 2, (agents - 1) * 2};
            neighborId = 0;
        }
        else if (a == agents - 1) {
            sa0.tag = {0, (a - 1) * 2, a * 2};
            neighborId = 2;
        }
        else {
            sa0.tag = {(a - 1) * 2, a * 2, (a + 1) * 2};
            neighborId = 1;
        }
        sa0.matrix = makeA0MatrixStatus(neighbors, neighborId, pFailBase, pFailBonus, pDeadBase, pDeadBonus);

        aif::DBN::Node sa1{{a * 2}, sa1Matrix};

        nodeStatus.nodes.emplace_back(std::move(sa0));
        nodeStatus.nodes.emplace_back(std::move(sa1));

        aif::FactoredDDN::Node nodeLoad{{a}, {}};

        // Here we only depend on our own previous load state
        aif::DBN::Node la0{{a * 2, (a * 2) + 1}, la0Matrix};
        aif::DBN::Node la1{{(a * 2) + 1}, la1Matrix};

        nodeLoad.nodes.emplace_back(std::move(la0));
        nodeLoad.nodes.emplace_back(std::move(la1));

        ddn.nodes.emplace_back(std::move(nodeStatus));
        ddn.nodes.emplace_back(std::move(nodeLoad));
    }

    // All reward matrices for all agents are the same, so we build it here
    // once.
    //
    // In particular, we get 1 reward each time we get to a Done state.
    // However, our matrix of rewards is SxA (with no end states), so we need
    // to convert our definition of reward into SxA format.
    //
    // This means that we need to see which dependencies the Load state has:
    // both the previous Load and previous Status.
    ai::Matrix2D rewardMatrix(3 * 3, 2);
    rewardMatrix.setZero();

    // Basically, the only way we can get reward is by:
    // - Starting from the Load state (since it's the only one that can complete)
    // - Doing action 0;
    // - And ending up in the Done state.
    //
    // Remember that R(s,a) = sum_s1 T(s,a,s1) * R(s,a,s1)
    rewardMatrix(Load * 3 + Good, 0) = la0Matrix(Load * 3 + Good, Done) * 1.0;
    rewardMatrix(Load * 3 + Fail, 0) = la0Matrix(Load * 3 + Fail, Done) * 1.0;
    rewardMatrix(Load * 3 + Dead, 0) = la0Matrix(Load * 3 + Dead, Done) * 1.0;

    aif::Factored2DMatrix rewards;
    for (size_t a = 0; a < agents; ++a) {
        // Now we set all of them with the correct dependencies.
        aif::BasisMatrix basis;
        basis.tag = {a * 2, a * 2 + 1}; // We depend on the before status and load;
        basis.actionTag = {a};   // And on our action.
        basis.values = rewardMatrix;

        rewards.bases.emplace_back(std::move(basis));
    }

    return afm::CooperativeModel(std::move(S), std::move(A), std::move(ddn), std::move(rewards));
}

afm::CooperativeModel makeSysAdminUniRing(unsigned agents,
    // Status transition params.
    double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus,
    // Load transition params.
    double pLoad, double pDoneG, double pDoneF)
{
    // Parameters for this network type:
    // In this ring we have 1 neighbor.
    constexpr unsigned neighbors = 1;

    // We factor the state space into two variables per each agent: status and
    // load. Each of them can assume 3 different values.
    aif::State S(agents * 2);
    std::fill(std::begin(S), std::end(S), 3);

    // Each agent has a single action, so the size of the action space is equal
    // to the number of agents.
    aif::Action A(agents);
    std::fill(std::begin(A), std::end(A), 2);

    // All matrices but the a0 status transitions do not depend on the
    // neighbors, so we can create them only once and just copy them when we
    // need them.
    const auto sa1Matrix = makeA1MatrixStatus();
    const auto la0Matrix = makeA0MatrixLoad(pLoad, pDoneG, pDoneF);
    const auto la1Matrix = makeA1MatrixStatus();

    auto ddn = aif::FactoredDDN();
    for (size_t a = 0; a < agents; ++a) {
        // Here, for each action, we have to create two transition nodes: one
        // for the status of the machine, and another for the load.
        // Both nodes only depend on the action of its agent.

        // Status node, only depends on the action of 'a'
        aif::FactoredDDN::Node nodeStatus{{a}, {}};

        // Status nodes for action 0 (do nothing) and action 1 (restart) respectively.
        // Node that the transition node for action 0 depends on the neighbors,
        // since whether they are failing or not affects whether this machine
        // will fail or not. If we reset, we don't really care.
        aif::DBN::Node sa0{{}, {}};
        unsigned neighborId;
        // Set the correct dependencies for the ring
        if (a == 0) {
            sa0.tag = {0, (agents - 1) * 2};
            neighborId = 0;
        }
        else {
            sa0.tag = {(a - 1) * 2, a * 2};
            neighborId = 1;
        }
        sa0.matrix = makeA0MatrixStatus(neighbors, neighborId, pFailBase, pFailBonus, pDeadBase, pDeadBonus);

        aif::DBN::Node sa1{{a * 2}, sa1Matrix};

        nodeStatus.nodes.emplace_back(std::move(sa0));
        nodeStatus.nodes.emplace_back(std::move(sa1));

        aif::FactoredDDN::Node nodeLoad{{a}, {}};

        // Here we only depend on our own previous load state
        aif::DBN::Node la0{{a * 2, (a * 2) + 1}, la0Matrix};
        aif::DBN::Node la1{{(a * 2) + 1}, la1Matrix};

        nodeLoad.nodes.emplace_back(std::move(la0));
        nodeLoad.nodes.emplace_back(std::move(la1));

        ddn.nodes.emplace_back(std::move(nodeStatus));
        ddn.nodes.emplace_back(std::move(nodeLoad));
    }

    // All reward matrices for all agents are the same, so we build it here
    // once.
    //
    // In particular, we get 1 reward each time we get to a Done state.
    // However, our matrix of rewards is SxA (with no end states), so we need
    // to convert our definition of reward into SxA format.
    //
    // This means that we need to see which dependencies the Load state has:
    // both the previous Load and previous Status.
    ai::Matrix2D rewardMatrix(3 * 3, 2);
    rewardMatrix.setZero();

    // Basically, the only way we can get reward is by:
    // - Starting from the Load state (since it's the only one that can complete)
    // - Doing action 0;
    // - And ending up in the Done state.
    //
    // Remember that R(s,a) = sum_s1 T(s,a,s1) * R(s,a,s1)
    rewardMatrix(Load * 3 + Good, 0) = la0Matrix(Load * 3 + Good, Done) * 1.0;
    rewardMatrix(Load * 3 + Fail, 0) = la0Matrix(Load * 3 + Fail, Done) * 1.0;
    rewardMatrix(Load * 3 + Dead, 0) = la0Matrix(Load * 3 + Dead, Done) * 1.0;

    aif::Factored2DMatrix rewards;
    for (size_t a = 0; a < agents; ++a) {
        // Now we set all of them with the correct dependencies.
        aif::BasisMatrix basis;
        basis.tag = {a * 2, a * 2 + 1}; // We depend on the before status and load;
        basis.actionTag = {a};   // And on our action.
        basis.values = rewardMatrix;

        rewards.bases.emplace_back(std::move(basis));
    }

    return afm::CooperativeModel(std::move(S), std::move(A), std::move(ddn), std::move(rewards));
}

unsigned ceil(unsigned x, unsigned y) {
    return (x + y - 1) / y;
}

char printMachineStatus(unsigned s) {
    switch (s) {
        case 0: return 'g';
        case 1: return 'f';
        default: return 'd';
    }
}

char printMachineLoad(unsigned l) {
    switch (l) {
        case 0: return 'i';
        case 1: return 'l';
        default: return 'd';
    }
}

std::string printSysAdminRing(const aif::State & s) {
    std::string retval;

    const size_t agents = s.size() / 2;

    const unsigned height = agents == 1 ? 1 : ceil(agents, 4) + 1;
    const unsigned width = agents == 1 ? 1 :
                           agents < 7 ? 2 :
                           ceil(agents - height * 2, 2) + 2;

    unsigned printRightId = 0;
    unsigned printLeftId = agents - 1;
    for (unsigned h = 0; h < height; ++h) {
        for (unsigned w = 0; w < width; ++w) {
            // Check if we need to print linkage or space
            if (w != 0 && (h == 0 || h == (height - 1)))
                retval += " -- ";
            else
                retval += "    ";

            // Check if we are in a printing spot
            if (h == 0 || h == (height - 1) ||
                w == 0 || w == (width - 1)) {
                // If we are, check that there's stuff to print
                if (agents != 1 && printLeftId == printRightId && w != width - 1) {
                    if (w == 0)
                        retval += "+-";
                    else
                        retval += "--";
                } else {
                    unsigned idToPrint;
                    if (h == 0 || w != 0)
                        idToPrint = printRightId++;
                    else
                        idToPrint = printLeftId--;
                    idToPrint *= 2;
                    // retval += '0' + idToPrint;
                    // retval += '0' + idToPrint;
                    retval += printMachineStatus(s[idToPrint]);
                    retval += printMachineLoad(s[idToPrint+1]);
                }
            // If we are not, fill with space
            } else {
                retval += "  ";
            }
        }
        retval += '\n';
        if (h != height - 1) {
            retval += "    | ";
            for (unsigned w = 0; w + 2 < width; ++w) {
                retval += "      ";
            }
            retval += "     |\n";
        }
    }

    return retval;
}

#endif
