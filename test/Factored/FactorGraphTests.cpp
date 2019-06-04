#define BOOST_TEST_MODULE Factored_FactorGraph
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/Utils/FactorGraph.hpp>

namespace aif = AIToolbox::Factored;

struct EmptyFactor {};

BOOST_AUTO_TEST_CASE( basic_construction ) {
    aif::FactorGraph<EmptyFactor> graph(15);

    BOOST_CHECK_EQUAL(graph.variableSize(), 15);
    BOOST_CHECK_EQUAL(graph.factorSize(), 0);
}

BOOST_AUTO_TEST_CASE( adding_rules ) {
    std::vector<aif::PartialKeys> rules {
        {0, 1}, // (1)
        {0, 2}, // (2)
        {0},    // (3)
        {2},    // (4)
    };

    const size_t agentsNum = 3;
    aif::FactorGraph<EmptyFactor> graph(agentsNum);
    for (const auto & rule : rules)
        graph.getFactor(rule);

    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum);
    BOOST_CHECK_EQUAL(graph.factorSize(), 4);

    BOOST_CHECK_EQUAL(graph.getFactors(0).size(), 3);
    BOOST_CHECK_EQUAL(graph.getFactors(1).size(), 1);
    BOOST_CHECK_EQUAL(graph.getFactors(2).size(), 2);

    BOOST_CHECK_EQUAL(graph.getVariables(0).size(), 2);
    BOOST_CHECK_EQUAL(graph.getVariables(1).size(), 1);
    BOOST_CHECK_EQUAL(graph.getVariables(2).size(), 1);
}

BOOST_AUTO_TEST_CASE( erase_agent ) {
    std::vector<aif::PartialKeys> rules {
        {0, 1}, // (1)
        {0, 2}, // (2)
        {0},    // (3)
        {2},    // (4)
        {1, 3}, // (5)
        {2, 3}, // (6)
        {0, 4}, // (7)
    };

    const size_t agentsNum = 5;
    aif::FactorGraph<EmptyFactor> graph(agentsNum);
    for (const auto & rule : rules)
        graph.getFactor(rule);

    auto cmp = [](const aif::PartialKeys & lhs, const aif::PartialKeys & rhs) {
        return AIToolbox::veccmp(lhs, rhs) == 0;
    };

    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum);
    BOOST_CHECK(cmp(graph.getVariables(0), {1, 2, 4}));
    BOOST_CHECK(cmp(graph.getVariables(1), {0, 3}));
    BOOST_CHECK(cmp(graph.getVariables(2), {0, 3}));
    BOOST_CHECK(cmp(graph.getVariables(3), {1, 2}));
    BOOST_CHECK(cmp(graph.getVariables(4), {0}));

    graph.erase(0);
    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum - 1);
    BOOST_CHECK(cmp(graph.getVariables(1), {3}));
    BOOST_CHECK(cmp(graph.getVariables(2), {3}));
    BOOST_CHECK(cmp(graph.getVariables(3), {1, 2}));
    BOOST_CHECK(cmp(graph.getVariables(4), {}));

    graph.erase(0);
    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum - 1);
    BOOST_CHECK(cmp(graph.getVariables(1), {3}));
    BOOST_CHECK(cmp(graph.getVariables(2), {3}));
    BOOST_CHECK(cmp(graph.getVariables(3), {1, 2}));
    BOOST_CHECK(cmp(graph.getVariables(4), {}));

    graph.erase(2);
    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum - 2);
    BOOST_CHECK(cmp(graph.getVariables(1), {3}));
    BOOST_CHECK(cmp(graph.getVariables(3), {1}));
    BOOST_CHECK(cmp(graph.getVariables(4), {}));

    graph.erase(4);
    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum - 3);
    BOOST_CHECK(cmp(graph.getVariables(1), {3}));
    BOOST_CHECK(cmp(graph.getVariables(3), {1}));

    graph.erase(3);
    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum - 4);
    BOOST_CHECK(cmp(graph.getVariables(1), {}));

    graph.erase(1);
    BOOST_CHECK_EQUAL(graph.variableSize(),  0);
}

BOOST_AUTO_TEST_CASE( erase_agent_2 ) {
    std::vector<aif::PartialKeys> rules {
        {0, 1}, // (1)
        {1, 2}, // (2)
        {0, 1, 2},    // (3)
    };

    const size_t agentsNum = 3;
    aif::FactorGraph<EmptyFactor> graph(agentsNum);
    for (const auto & rule : rules)
        graph.getFactor(rule);

    auto cmp = [](const aif::PartialKeys & lhs, const aif::PartialKeys & rhs) {
        return AIToolbox::veccmp(lhs, rhs) == 0;
    };

    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum);
    BOOST_CHECK(cmp(graph.getVariables(0), {1, 2}));
    BOOST_CHECK(cmp(graph.getVariables(1), {0, 2}));
    BOOST_CHECK(cmp(graph.getVariables(2), {0, 1}));

    graph.erase(0);
    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum - 1);
    BOOST_CHECK(cmp(graph.getVariables(1), {2}));
    BOOST_CHECK(cmp(graph.getVariables(2), {1}));

    graph.erase(1);
    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum - 2);
    BOOST_CHECK(cmp(graph.getVariables(2), {}));

    graph.erase(2);
    BOOST_CHECK_EQUAL(graph.variableSize(),  agentsNum - 3);
}

BOOST_AUTO_TEST_CASE( neighbors ) {
    std::vector<aif::PartialKeys> rules {
        {0},
        {0, 1},
        {0, 2},
        {0, 3},
        {0, 4},
    };

    const size_t agentsNum = 5;
    aif::FactorGraph<EmptyFactor> graph(agentsNum);
    for (const auto & rule : rules)
        graph.getFactor(rule);

    auto f = graph.getFactors(0);
    BOOST_CHECK_EQUAL(f.size(), 5);

    auto a = graph.getVariables(f);
    BOOST_CHECK_EQUAL(a.size(), 5);
}

BOOST_AUTO_TEST_CASE( best_removal ) {
    std::vector<aif::PartialKeys> rules {
        {0, 1}, // (1)
        {0, 2}, // (2)
        {0},    // (3)
        {2},    // (4)
        {1, 3}, // (5)
        {2, 3}, // (6)
        {0, 4}, // (7)
    };

    const size_t agentsNum = 5;
    aif::FactorGraph<EmptyFactor> graph(agentsNum);
    for (const auto & rule : rules)
        graph.getFactor(rule);

    const aif::Factors F = {2, 3, 4, 3, 3};

    auto a = graph.bestVariableToRemove(F);
    BOOST_CHECK_EQUAL(a, 4);
    graph.erase(a);

    a = graph.bestVariableToRemove(F);
    BOOST_CHECK_EQUAL(a, 1);
    graph.erase(a);

    a = graph.bestVariableToRemove(F);
    BOOST_CHECK_EQUAL(a, 0);
    graph.erase(a);

    a = graph.bestVariableToRemove(F);
    BOOST_CHECK(a == 2 || a == 3);
    graph.erase(a);

    auto a2 = graph.bestVariableToRemove(F);
    BOOST_CHECK(a != a2 && (a2 == 2 || a2 == 3));
    graph.erase(a);
}
