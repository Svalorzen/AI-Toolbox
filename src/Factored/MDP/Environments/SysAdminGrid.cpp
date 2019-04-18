#include <AIToolbox/Factored/MDP/Environments/SysAdmin.hpp>

#include "./SysAdminUtils.hpp"
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/MDP/Environments/Utils/GridWorld.hpp>

#include <algorithm>

namespace AIToolbox::Factored::MDP {
    CooperativeModel makeSysAdminGrid(unsigned width, unsigned height,
        // Status transition params.
        double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus,
        // Load transition params.
        double pLoad, double pDoneG, double pDoneF)
    {
        using namespace SysAdminEnums;

        // Make grid world to help with directions.
        AIToolbox::MDP::GridWorld grid(width, height);
        const auto agents = grid.getS();

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
            auto cell = grid(a);
            sa0.tag = {cell * 2};
            // Add to tag all elements around it.
            for (auto d : AIToolbox::MDP::GridWorldEnums::Directions) {
                const auto adj = grid.getAdjacent(d, cell);
                if (adj == cell) continue;
                sa0.tag.push_back(grid.getAdjacent(d, cell) * 2);
            }

            // Sort them so the tag is valid.
            auto beg = std::begin(sa0.tag), end = std::end(sa0.tag);
            std::sort(beg, end);

            // Find out where we are in the tag so we can generate the matrix correctly.
            unsigned neighborId = std::distance(beg, std::find(beg, end, cell * 2));
            sa0.matrix = makeA0MatrixStatus(sa0.tag.size() - 1, neighborId, pFailBase, pFailBonus, pDeadBase, pDeadBonus);

            DBN::Node sa1{{cell * 2}, sa1Matrix};

            nodeStatus.nodes.emplace_back(std::move(sa0));
            nodeStatus.nodes.emplace_back(std::move(sa1));

            FactoredDDN::Node nodeLoad{{a}, {}};

            // Here we only depend on our own previous load state
            DBN::Node la0{{cell * 2, (cell * 2) + 1}, la0Matrix};
            DBN::Node la1{{(cell * 2) + 1}, la1Matrix};

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

    CooperativeModel makeSysAdminTorus(unsigned width, unsigned height,
        // Status transition params.
        double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus,
        // Load transition params.
        double pLoad, double pDoneG, double pDoneF)
    {
        using namespace SysAdminEnums;
        // Parameters for this network type:
        // Since we are using a torus, each agent has 4 neighbors.
        constexpr unsigned neighbors = 4;

        // Make torus grid world to help with directions.
        AIToolbox::MDP::GridWorld grid(width, height, true);
        const auto agents = grid.getS();

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
            auto cell = grid(a);
            sa0.tag = {cell * 2};
            // Add to tag all elements around it.
            for (auto d : AIToolbox::MDP::GridWorldEnums::Directions)
                sa0.tag.push_back(grid.getAdjacent(d, cell) * 2);

            // Sort them so the tag is valid.
            auto beg = std::begin(sa0.tag), end = std::end(sa0.tag);
            std::sort(beg, end);

            // Find out where we are in the tag so we can generate the matrix correctly.
            unsigned neighborId = std::distance(beg, std::find(beg, end, cell * 2));
            sa0.matrix = makeA0MatrixStatus(neighbors, neighborId, pFailBase, pFailBonus, pDeadBase, pDeadBonus);

            DBN::Node sa1{{cell * 2}, sa1Matrix};

            nodeStatus.nodes.emplace_back(std::move(sa0));
            nodeStatus.nodes.emplace_back(std::move(sa1));

            FactoredDDN::Node nodeLoad{{a}, {}};

            // Here we only depend on our own previous load state
            DBN::Node la0{{cell * 2, (cell * 2) + 1}, la0Matrix};
            DBN::Node la1{{(cell * 2) + 1}, la1Matrix};

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

    std::string printSysAdminGrid(const State & s, const unsigned width) {
        std::string retval;

        const size_t agents = s.size() / 2;

        const unsigned height = agents / width;

        AIToolbox::MDP::GridWorld grid(width, height);

        for (unsigned h = 0; h < height; ++h) {
            for (unsigned w = 0; w < width; ++w) {
                if (w > 0)
                    retval += " -- ";

                retval += printMachineStatus(s[grid(w, h) * 2]);
                retval += printMachineLoad(s[grid(w,h) * 2 + 1]);
            }
            retval += '\n';
            if (h < height - 1) {
                retval += "| ";
                for (unsigned w = 1; w < width; ++w) {
                    retval += "     |";
                }
                retval += '\n';
            }
        }
        return retval;
    }
}
