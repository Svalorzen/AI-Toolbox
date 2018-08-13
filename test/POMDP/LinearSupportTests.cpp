#define BOOST_TEST_MODULE POMDP_Incremental_Pruning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/LinearSupport.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/Utils/Core.hpp>

#include "Utils/TigerProblem.hpp"

BOOST_AUTO_TEST_CASE( vertex_enumeration ) {
    using namespace AIToolbox;

    POMDP::VList alphas = {
        {(MDP::Values(3) << 1.0, 0.0, 0.0).finished(), 0, POMDP::VObs(0)},
        {(MDP::Values(3) << 0.0, 1.0, 0.0).finished(), 0, POMDP::VObs(0)},
        {(MDP::Values(3) << 0.0, 0.0, 1.0).finished(), 0, POMDP::VObs(0)},
    };

    std::vector<std::pair<POMDP::Belief, double>> solutions = {
        {(POMDP::Belief(3) << 1.0/3.0, 1.0/3.0, 1.0/3.0).finished(), 1.0/3.0},
        {(POMDP::Belief(3) << 0.5, 0.5, 0.0).finished(), 0.5},
        {(POMDP::Belief(3) << 0.0, 0.5, 0.5).finished(), 0.5},
        {(POMDP::Belief(3) << 0.5, 0.0, 0.5).finished(), 0.5},
    };

    // We look for all vertices from all possible angles.  We are going to get
    // duplicates for now, but that's not a problem as long as all vertices are
    // enumerated.
    auto vertices = POMDP::findVertices(std::begin(alphas), std::end(alphas), std::begin(alphas), std::end(alphas));

    // Now we check against the solution, both ways: all vertices in the
    // solution must be somewhere in the new list, and all vertices in the new
    // list must be in the solution.
    for (const auto & v : vertices) {
        bool found = false;
        for (const auto & s : solutions) {
            if (v.first == s.first && v.second == s.second) {
                found = true;
                break;
            }
        }
        BOOST_CHECK(found);
    }

    for (const auto & s : solutions) {
        bool found = false;
        for (const auto & v : vertices) {
            if (v.first == s.first && v.second == s.second) {
                found = true;
                break;
            }
        }
        BOOST_CHECK(found);
    }
}
