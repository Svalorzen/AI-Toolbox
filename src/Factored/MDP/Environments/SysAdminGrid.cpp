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
        using namespace SysAdminUtils;

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

        DDNGraph graph(S, A);

        // All matrices but the a0 status transitions do not depend on the
        // neighbors, so we can create them only once and just copy them when we
        // need them.
        const auto sa1Matrix = makeA1MatrixStatus();
        const auto la0Matrix = makeA0MatrixLoad(pLoad, pDoneG, pDoneF);
        const auto la1Matrix = makeA1MatrixLoad();

        auto transitions = DDN::TransitionMatrix();

        for (size_t a = 0; a < agents; ++a) {
            // Here, for each action, we have to create two transition nodes: one
            // for the status of the machine, and another for the load.
            // Both nodes only depend on the action of its agent.

            // Status node, only depends on the action of 'a'
            DDNGraph::ParentSet statusParents{{a}, {}};

            // Status nodes for action 0 (do nothing) and action 1 (restart) respectively.
            // Note that the transition node for action 0 depends on the neighbors,
            // since whether they are failing or not affects whether this machine
            // will fail or not. If we reset, we don't really care.
            auto cell = grid(a);
            PartialKeys sa0 = {cell * 2};
            // Add to tag all elements around it.
            for (auto d : AIToolbox::MDP::GridWorldUtils::Directions4) {
                const auto adj = grid.getAdjacent(d, cell);
                if (adj == cell) continue;
                sa0.push_back(grid.getAdjacent(d, cell) * 2);
            }

            // Sort them so the tag is valid.
            auto beg = std::begin(sa0), end = std::end(sa0);
            std::sort(beg, end);

            statusParents.features.push_back(sa0);
            statusParents.features.push_back({cell*2});

            graph.push(std::move(statusParents));

            transitions.emplace_back(graph.getSize(cell * 2), S[cell * 2]);
            {
                auto & T = transitions.back();

                // Find out where we are in the tag so we can generate the matrix correctly.
                unsigned neighborId = std::distance(beg, std::find(beg, end, cell * 2));
                T.topRows(T.rows() - sa1Matrix.rows()) = makeA0MatrixStatus(sa0.size() - 1, neighborId, pFailBase, pFailBonus, pDeadBase, pDeadBonus);
                T.bottomRows(sa1Matrix.rows()) = sa1Matrix;
            }

            DDNGraph::ParentSet loadParents{{a}, {}};

            loadParents.features.push_back({cell * 2, (cell * 2) + 1});
            loadParents.features.push_back({(cell * 2) + 1});

            graph.push(std::move(loadParents));

            transitions.emplace_back(graph.getSize(cell * 2 + 1), S[cell * 2 + 1]);
            {
                auto & T = transitions.back();

                T.topRows(la0Matrix.rows()) = la0Matrix;
                T.bottomRows(la1Matrix.rows()) = la1Matrix;
            }
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
        return CooperativeModel(std::move(graph), std::move(transitions), std::move(rewards), 0.95);
    }

    CooperativeModel makeSysAdminTorus(unsigned width, unsigned height,
        // Status transition params.
        double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus,
        // Load transition params.
        double pLoad, double pDoneG, double pDoneF)
    {
        using namespace SysAdminUtils;
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

        DDNGraph graph(S, A);

        // All matrices but the a0 status transitions do not depend on the
        // neighbors, so we can create them only once and just copy them when we
        // need them.
        const auto sa1Matrix = makeA1MatrixStatus();
        const auto la0Matrix = makeA0MatrixLoad(pLoad, pDoneG, pDoneF);
        const auto la1Matrix = makeA1MatrixLoad();

        auto transitions = DDN::TransitionMatrix();

        for (size_t a = 0; a < agents; ++a) {
            // Here, for each action, we have to create two transition nodes: one
            // for the status of the machine, and another for the load.
            // Both nodes only depend on the action of its agent.

            // Status node, only depends on the action of 'a'
            DDNGraph::ParentSet statusParents{{a}, {}};

            // Status nodes for action 0 (do nothing) and action 1 (restart) respectively.
            // Note that the transition node for action 0 depends on the neighbors,
            // since whether they are failing or not affects whether this machine
            // will fail or not. If we reset, we don't really care.
            auto cell = grid(a);
            PartialKeys sa0 = {cell * 2};
            // Add to tag all elements around it.
            for (auto d : AIToolbox::MDP::GridWorldUtils::Directions4)
                sa0.push_back(grid.getAdjacent(d, cell) * 2);

            // Sort them so the tag is valid.
            auto beg = std::begin(sa0), end = std::end(sa0);
            std::sort(beg, end);

            statusParents.features.push_back(sa0);
            statusParents.features.push_back({cell * 2});

            graph.push(std::move(statusParents));

            transitions.emplace_back(graph.getSize(cell * 2), S[cell * 2]);
            {
                auto & T = transitions.back();

                // Find out where we are in the tag so we can generate the matrix correctly.
                unsigned neighborId = std::distance(beg, std::find(beg, end, cell * 2));
                T.topRows(T.rows() - sa1Matrix.rows()) = makeA0MatrixStatus(neighbors, neighborId, pFailBase, pFailBonus, pDeadBase, pDeadBonus);
                T.bottomRows(sa1Matrix.rows()) = sa1Matrix;
            }

            DDNGraph::ParentSet loadParents{{a}, {}};

            loadParents.features.push_back({cell * 2, (cell * 2) + 1});
            loadParents.features.push_back({(cell * 2) + 1});

            graph.push(std::move(loadParents));

            transitions.emplace_back(graph.getSize(cell * 2 + 1), S[cell * 2 + 1]);
            {
                auto & T = transitions.back();

                // Here we only depend on our own previous load state
                T.topRows(la0Matrix.rows()) = la0Matrix;
                T.bottomRows(la1Matrix.rows()) = la1Matrix;
            }
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
        return CooperativeModel(std::move(graph), std::move(transitions), std::move(rewards), 0.95);
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
