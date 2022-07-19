#define BOOST_TEST_MODULE Factored_FactorGraph
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "GlobalFixtures.hpp"

#include <AIToolbox/Factored/Utils/FactorGraph.hpp>
#include <AIToolbox/Factored/Utils/APSP.hpp>

namespace aif = AIToolbox::Factored;

struct EmptyFactor {};
struct IntFactor { int v; };

BOOST_AUTO_TEST_CASE( basic_construction ) {
    aif::FactorGraph<EmptyFactor> graph(15);

    BOOST_CHECK_EQUAL(graph.variableSize(), 15);
    BOOST_CHECK_EQUAL(graph.factorSize(), 0);
}

BOOST_AUTO_TEST_CASE( copy_construction ) {
    std::vector<aif::PartialKeys> rules {
        {0, 1}, // (1)
        {0, 2}, // (2)
        {0},    // (3)
        {2},    // (4)
    };

    const size_t agentsNum = 3;
    aif::FactorGraph<IntFactor> graph(agentsNum);
    int counter = 0;
    for (const auto & rule : rules) {
        auto & f = graph.getFactor(rule)->getData();
        f.v = ++counter;
    }

    auto graphCopy = graph;

    // Check everything is the same
    BOOST_CHECK_EQUAL(graph.factorSize(), graphCopy.factorSize());
    BOOST_CHECK_EQUAL(graph.variableSize(), graphCopy.variableSize());

    // Save info about original graph for later
    const auto factorSize = graph.factorSize();
    const auto variableSize = graph.variableSize();

    std::vector<decltype(graph)::FactorNode> fCopied;
    std::vector<std::pair<
        decltype(graph)::Variables,
        decltype(graph)::FactorItList
    >> vCopied;

    // Check factors contents
    {
        auto it = graph.begin();
        auto itc = graphCopy.begin();
        for (size_t i = 0; i < graph.factorSize(); ++i) {
            // Check factors correspond to same variables
            BOOST_CHECK(it->getVariables() == itc->getVariables());
            // Check data is the same.
            BOOST_CHECK_EQUAL(it->getData().v, itc->getData().v);

            fCopied.push_back(*it);

            ++it; ++itc;
        }
    }

    // Check variable contents
    for (size_t i = 0; i < graph.variableSize(); ++i) {
        BOOST_CHECK(graph.getVariables(i) == graphCopy.getVariables(i));

        // The factors sizes should be equal, but the contents must NOT be
        // equal, as they are iterators; thus pointers. Each graph should
        // point to itself only.
        auto factors = graph.getFactors(i);
        auto factorsCopy = graphCopy.getFactors(i);

        BOOST_CHECK_EQUAL(factors.size(), factorsCopy.size());

        // This is not foolproof as they *could* be shuffled in theory, but
        // hopefully the copy-constructor is not written in such a way that
        // shuffling is possible.
        for (size_t j = 0; j < factors.size(); ++j)
            BOOST_CHECK(factors[j] != factorsCopy[j]);

        vCopied.emplace_back(graph.getVariables(i), factors);
    }

    // Remove everything from copy graph.
    graphCopy.erase(0);
    graphCopy.erase(1);
    graphCopy.erase(2);

    // Check that original graph is still there (same checks as before, but
    // with the saved variables).
    BOOST_CHECK_EQUAL(factorSize, graph.factorSize());
    BOOST_CHECK_EQUAL(variableSize, graph.variableSize());
    {
        auto it = fCopied.begin();
        auto itc = graph.begin();
        for (size_t i = 0; i < factorSize; ++i) {
            // Check factors correspond to same variables
            BOOST_CHECK(it->getVariables() == itc->getVariables());
            // Check data is the same.
            BOOST_CHECK_EQUAL(it->getData().v, itc->getData().v);

            ++it; ++itc;
        }
    }

    // Check variable contents
    for (size_t i = 0; i < variableSize; ++i) {
        BOOST_CHECK(vCopied[i].first == graph.getVariables(i));
        BOOST_CHECK(vCopied[i].second == graph.getFactors(i));
    }
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

BOOST_AUTO_TEST_CASE( factor_order_correct ) {
    std::vector<aif::PartialKeys> rules {
        {0, 1}, // (1)
        {0, 2}, // (2)
        {0},    // (3)
        {2},    // (4)
        {1, 3}, // (5)
        {2, 3}, // (6)
        {0, 4}, // (7)
    };

    {
        const size_t agentsNum = 5;
        aif::FactorGraph<EmptyFactor> graph(agentsNum);
        for (const auto & rule : rules)
            graph.getFactor(rule);

        size_t i = 0;
        for (const auto & f : graph)
            BOOST_CHECK(f.getVariables() == rules[i++]);
    }
    // Now we do it in reverse; also to make sure that there's stuff in the
    // pool so that branch gets used.
    {
        const size_t agentsNum = 5;
        aif::FactorGraph<EmptyFactor> graph(agentsNum);
        for (auto rit = rules.rbegin(); rit != rules.rend(); ++rit)
            graph.getFactor(*rit);

        size_t i = rules.size();
        for (const auto & f : graph)
            BOOST_CHECK(f.getVariables() == rules[--i]);
    }
}

BOOST_AUTO_TEST_CASE( small_graph_diameter ) {
    aif::FactorGraph<EmptyFactor> graph(4);

    // ###########
    // #         #
    // #    O    #
    // #   / \   #
    // #  /   \  #
    // # O-----O #
    // #  \   /  #
    // #   \ /   #
    // #    O    #
    // #         #
    // ###########

    graph.getFactor({0, 1, 2});
    graph.getFactor({1, 2, 3});

    BOOST_CHECK_EQUAL(APSP(graph), 2);
}

BOOST_AUTO_TEST_CASE( medium_graph_diameter ) {
    aif::FactorGraph<EmptyFactor> graph(7);

    // ###############
    // #             #
    // #      O      #
    // #     / \     #
    // # O--O   O--O #
    // #     \ /     #
    // #      O--O   #
    // #             #
    // ###############

    graph.getFactor({0, 1});
    graph.getFactor({1, 2});
    graph.getFactor({1, 3});
    graph.getFactor({3, 4});
    graph.getFactor({2, 5});
    graph.getFactor({3, 5});
    graph.getFactor({5, 6});

    BOOST_CHECK_EQUAL(APSP(graph), 4);
}

BOOST_AUTO_TEST_CASE( disjoint_graph_diameter ) {
    aif::FactorGraph<EmptyFactor> graph(3);

    graph.getFactor({0});
    graph.getFactor({1});
    graph.getFactor({2});

    BOOST_CHECK_EQUAL(APSP(graph), 0);
}
