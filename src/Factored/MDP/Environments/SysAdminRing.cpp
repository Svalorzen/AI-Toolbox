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
        using namespace SysAdminUtils;
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

        auto graph = DDNGraph(S, A);

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

            // ----- Status ------

            DDNGraph::ParentSet statusParents{{a}, {}};

            // Status nodes for action 0 (do nothing) and action 1 (restart) respectively.
            // Note that the transition node for action 0 depends on the neighbors,
            // since whether they are failing or not affects whether this machine
            // will fail or not. If we reset, we don't really care.
            unsigned neighborId;
            // Set the correct dependencies for the ring
            if (a == 0) {
                statusParents.features.push_back({0, (agents - 1) * 2});
                neighborId = 0;
            }
            else {
                statusParents.features.push_back({(a - 1) * 2, a * 2});
                neighborId = 1;
            }
            statusParents.features.push_back({a * 2});
            graph.push(std::move(statusParents));

            transitions.emplace_back(graph.getSize(a * 2), S[a * 2]);
            {
                auto & T = transitions.back();

                T.topRows(T.rows() - sa1Matrix.rows()) = makeA0MatrixStatus(neighbors, neighborId, pFailBase, pFailBonus, pDeadBase, pDeadBonus);
                T.bottomRows(sa1Matrix.rows()) = sa1Matrix;
            }

            // ----- Load ------

            DDNGraph::ParentSet loadParents{{a}, {}};

            // Here we only depend on our own previous load state
            loadParents.features.push_back({a * 2, (a * 2) + 1});
            loadParents.features.push_back({(a * 2) + 1});

            graph.push(std::move(loadParents));

            transitions.emplace_back(graph.getSize(a * 2 + 1), S[a * 2 + 1]);
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

    CooperativeModel makeSysAdminBiRing(unsigned agents,
        // Status transition params.
        double pFailBase, double pFailBonus, double pDeadBase, double pDeadBonus,
        // Load transition params.
        double pLoad, double pDoneG, double pDoneF)
    {
        using namespace SysAdminUtils;
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

        auto graph = DDNGraph(S, A);

        // All matrices but the a0 status transitions do not depend on the
        // neighbors, so we can create them only once and just copy them when we
        // need them.
        const auto sa1Matrix = makeA1MatrixStatus();
        const auto la0Matrix = makeA0MatrixLoad(pLoad, pDoneG, pDoneF);
        const auto la1Matrix = makeA1MatrixLoad();

        auto transitions = DDN::TransitionMatrix();

        for (size_t a = 0; a < agents; ++a) {
            // Status dependencies
            // Here, for each action, we have to create two transition nodes: one
            // for the status of the machine, and another for the load.
            // Both nodes only depend on the action of its agent.

            // ----- Status ------

            DDNGraph::ParentSet nodeStatus{{a}, {}};

            // Status nodes for action 0 (do nothing) and action 1 (restart) respectively.
            // Note that the transition node for action 0 depends on the neighbors,
            // since whether they are failing or not affects whether this machine
            // will fail or not. If we reset, we don't really care.
            unsigned neighborId;
            // Set the correct dependencies for the ring
            if (a == 0) {
                nodeStatus.features.push_back({0, 2, (agents - 1) * 2});
                neighborId = 0;
            }
            else if (a == agents - 1) {
                nodeStatus.features.push_back({0, (a - 1) * 2, a * 2});
                neighborId = 2;
            }
            else {
                nodeStatus.features.push_back({(a - 1) * 2, a * 2, (a + 1) * 2});
                neighborId = 1;
            }
            nodeStatus.features.push_back({a*2});
            graph.push(std::move(nodeStatus));

            transitions.emplace_back(graph.getSize(a * 2), S[a * 2]);
            {
                auto & T = transitions.back();

                T.topRows(T.rows() - sa1Matrix.rows()) = makeA0MatrixStatus(neighbors, neighborId, pFailBase, pFailBonus, pDeadBase, pDeadBonus);
                T.bottomRows(sa1Matrix.rows()) = sa1Matrix;
            }

            // ----- Load ------

            DDNGraph::ParentSet nodeLoad{{a}, {}};
            nodeLoad.features.push_back({a * 2, (a * 2) + 1});
            nodeLoad.features.push_back({(a * 2) + 1});

            graph.push(std::move(nodeLoad));

            transitions.emplace_back(graph.getSize(a * 2 + 1), S[a * 2 + 1]);
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
