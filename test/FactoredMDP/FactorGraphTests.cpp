#define BOOST_TEST_MODULE FactoredMDP_FactorGraph
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/FactoredMDP/FactorGraph.hpp>

namespace fm = AIToolbox::FactoredMDP;

struct EmptyFactor {};

BOOST_AUTO_TEST_CASE( basic_construction ) {
    fm::FactorGraph<EmptyFactor> graph(15);

    BOOST_CHECK_EQUAL(graph.agentSize(), 15);
    BOOST_CHECK_EQUAL(graph.factorSize(), 0);
}

BOOST_AUTO_TEST_CASE( adding_rules ) {
    std::vector<fm::PartialAction> rules {
        {{0, 1}, {1, 2}}, // (1)
        {{0, 2}, {1, 2}}, // (2)
        {{0, 1}, {0, 1}}, // (1)
        {{0, 1}, {2, 2}}, // (1)
        {{0, 1}, {4, 4}}, // (1)
        {{0, 1}, {2, 9}}, // (1)
        {{0, 2}, {1, 3}}, // (2)
        {{0},    {0}},    // (3)
        {{2},    {0}},    // (4)
    };

    const size_t agentsNum = 3;
    fm::FactorGraph<EmptyFactor> graph(agentsNum);
    for (const auto & rule : rules)
        graph.getFactor(rule.first);

    BOOST_CHECK_EQUAL(graph.agentSize(),  agentsNum);
    BOOST_CHECK_EQUAL(graph.factorSize(), 4);

    BOOST_CHECK_EQUAL(graph.getNeighbors(0).size(), 3);
    BOOST_CHECK_EQUAL(graph.getNeighbors(1).size(), 1);
    BOOST_CHECK_EQUAL(graph.getNeighbors(2).size(), 2);
}

BOOST_AUTO_TEST_CASE( erase_factor ) {
    std::vector<size_t> rule{0, 1};

    const size_t agentsNum = 3;
    fm::FactorGraph<EmptyFactor> graph(agentsNum);
    for (unsigned i = 0; i < 10; ++i)
        graph.getFactor(rule);
    // We only added it once
    BOOST_CHECK_EQUAL(graph.factorSize(), 1);

    auto it = graph.getFactor(rule);
    graph.erase(it);
    // After erasing it, we should have no factors.
    BOOST_CHECK_EQUAL(graph.factorSize(), 0);
    // Re-inserting it should work.
    graph.getFactor(rule);
    BOOST_CHECK_EQUAL(graph.factorSize(), 1);
}

BOOST_AUTO_TEST_CASE( erase_agent ) {
    const size_t agentsNum = 3;
    fm::FactorGraph<EmptyFactor> graph(agentsNum);

    BOOST_CHECK_EQUAL(graph.agentSize(),  agentsNum);
    graph.erase(0);
    BOOST_CHECK_EQUAL(graph.agentSize(),  agentsNum - 1);
    graph.erase(1);
    graph.erase(2);
    BOOST_CHECK_EQUAL(graph.agentSize(),  0);
}
