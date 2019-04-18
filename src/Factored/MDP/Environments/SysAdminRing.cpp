#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

#include "./SysAdminUtils.hpp"
#include <AIToolbox/Utils/Core.hpp>

#include <algorithm>

namespace AIToolbox::Factored::MDP {
    CooperativeModel makeSysAdminUniRing(unsigned agents,
        // Status transition params.
        double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus,
        // Load transition params.
        double pLoad, double pDoneG, double pDoneF)
    {
        using namespace SysAdminEnums;
        // Parameters for this network type:
        // In this ring we have 1 neighbor.
        constexpr unsigned neighbors = 1;

        // We factor the state space into two variables per each agent: status and
        // load. Each of them can assume 3 different values.
        State S(agents * 2);
        std::fill(std::begin(S), std::end(S), 3);

        // Each agent has a single action, so the size of the action space is equal
        // to the number of agents.
        Action A(agents);
        std::fill(std::begin(A), std::end(A), 2);

        // All matrices but the a0 status transitions do not depend on the
        // neighbors, so we can create them only once and just copy them when we
        // need them.
        const auto sa1Matrix = makeA1MatrixStatus();
        const auto la0Matrix = makeA0MatrixLoad(pLoad, pDoneG, pDoneF);
        const auto la1Matrix = makeA1MatrixLoad();

        auto ddn = FactoredDDN();
        for (size_t a = 0; a < agents; ++a) {
            // Here, for each action, we have to create two transition nodes: one
            // for the status of the machine, and another for the load.
            // Both nodes only depend on the action of its agent.

            // Status node, only depends on the action of 'a'
            FactoredDDN::Node nodeStatus{{a}, {}};

            // Status nodes for action 0 (do nothing) and action 1 (restart) respectively.
            // Node that the transition node for action 0 depends on the neighbors,
            // since whether they are failing or not affects whether this machine
            // will fail or not. If we reset, we don't really care.
            DBN::Node sa0{{}, {}};
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

            DBN::Node sa1{{a * 2}, sa1Matrix};

            nodeStatus.nodes.emplace_back(std::move(sa0));
            nodeStatus.nodes.emplace_back(std::move(sa1));

            FactoredDDN::Node nodeLoad{{a}, {}};

            // Here we only depend on our own previous load state
            DBN::Node la0{{a * 2, (a * 2) + 1}, la0Matrix};
            DBN::Node la1{{(a * 2) + 1}, la1Matrix};

            nodeLoad.nodes.emplace_back(std::move(la0));
            nodeLoad.nodes.emplace_back(std::move(la1));

            ddn.nodes.emplace_back(std::move(nodeStatus));
            ddn.nodes.emplace_back(std::move(nodeLoad));
        }

        // All reward matrices for all agents are the same, so we build it here
        // once.
        Matrix2D rewardMatrix = makeRewardMatrix(la0Matrix);

        FactoredMatrix2D rewards;
        for (size_t a = 0; a < agents; ++a) {
            // Now we set all of them with the correct dependencies.
            BasisMatrix basis;
            basis.tag = {a * 2, a * 2 + 1}; // We depend on the before status and load;
            basis.actionTag = {a};   // And on our action.
            basis.values = rewardMatrix;

            rewards.bases.emplace_back(std::move(basis));
        }

        return CooperativeModel(std::move(S), std::move(A), std::move(ddn), std::move(rewards), 0.95);
    }

    CooperativeModel makeSysAdminBiRing(unsigned agents,
        // Status transition params.
        double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus,
        // Load transition params.
        double pLoad, double pDoneG, double pDoneF)
    {
        using namespace SysAdminEnums;
        // Parameters for this network type:
        // In a ring we have 2 neighbors.
        constexpr unsigned neighbors = 2;

        // We factor the state space into two variables per each agent: status and
        // load. Each of them can assume 3 different values.
        State S(agents * 2);
        std::fill(std::begin(S), std::end(S), 3);

        // Each agent has a single action, so the size of the action space is equal
        // to the number of agents.
        Action A(agents);
        std::fill(std::begin(A), std::end(A), 2);

        // All matrices but the a0 status transitions do not depend on the
        // neighbors, so we can create them only once and just copy them when we
        // need them.
        const auto sa1Matrix = makeA1MatrixStatus();
        const auto la0Matrix = makeA0MatrixLoad(pLoad, pDoneG, pDoneF);
        const auto la1Matrix = makeA1MatrixLoad();

        auto ddn = FactoredDDN();
        for (size_t a = 0; a < agents; ++a) {
            // Here, for each action, we have to create two transition nodes: one
            // for the status of the machine, and another for the load.
            // Both nodes only depend on the action of its agent.

            // Status node, only depends on the action of 'a'
            FactoredDDN::Node nodeStatus{{a}, {}};

            // Status nodes for action 0 (do nothing) and action 1 (restart) respectively.
            // Node that the transition node for action 0 depends on the neighbors,
            // since whether they are failing or not affects whether this machine
            // will fail or not. If we reset, we don't really care.
            DBN::Node sa0{{}, {}};
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

            DBN::Node sa1{{a * 2}, sa1Matrix};

            nodeStatus.nodes.emplace_back(std::move(sa0));
            nodeStatus.nodes.emplace_back(std::move(sa1));

            FactoredDDN::Node nodeLoad{{a}, {}};

            // Here we only depend on our own previous load state
            DBN::Node la0{{a * 2, (a * 2) + 1}, la0Matrix};
            DBN::Node la1{{(a * 2) + 1}, la1Matrix};

            nodeLoad.nodes.emplace_back(std::move(la0));
            nodeLoad.nodes.emplace_back(std::move(la1));

            ddn.nodes.emplace_back(std::move(nodeStatus));
            ddn.nodes.emplace_back(std::move(nodeLoad));
        }

        // All reward matrices for all agents are the same, so we build it here
        // once.
        Matrix2D rewardMatrix = makeRewardMatrix(la0Matrix);

        FactoredMatrix2D rewards;
        for (size_t a = 0; a < agents; ++a) {
            // Now we set all of them with the correct dependencies.
            BasisMatrix basis;
            basis.tag = {a * 2, a * 2 + 1}; // We depend on the before status and load;
            basis.actionTag = {a};   // And on our action.
            basis.values = rewardMatrix;

            rewards.bases.emplace_back(std::move(basis));
        }

        return CooperativeModel(std::move(S), std::move(A), std::move(ddn), std::move(rewards), 0.95);
    }

    std::string printSysAdminRing(const State & s) {
        std::string retval;

        const size_t agents = s.size() / 2;

        const unsigned height = agents == 1 ? 1 : ceil(agents, 4) + 1;
        const unsigned width = agents < 3 ? 1 :
                               agents < 6 ? 2 :
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
                if (width > 1)
                    retval += "     |";
                retval += '\n';
            }
        }
        return retval;
    }
}
