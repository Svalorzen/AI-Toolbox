#define BOOST_TEST_MODULE FactoredMDP_MultiObjectiveVariableElimination
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <AIToolbox/FactoredMDP/Algorithms/Utils/MultiObjectiveVariableElimination.hpp>

namespace fm = AIToolbox::FactoredMDP;
using MOVE = fm::MultiObjectiveVariableElimination;

BOOST_AUTO_TEST_CASE( simple_graph ) {
    std::vector<fm::MOQFunctionRule> rules {
        // States, Actions,                     Value
        {    {},   {{0}, {0}},            (fm::Rewards(2) << 4.0, 0,0).finished()},
        {    {},   {{1}, {0}},            (fm::Rewards(2) << 5.0, 1.0).finished()},
        {    {},   {{1}, {1}},            (fm::Rewards(2) << 2.0, 2.0).finished()},
        {    {},   {{0, 1}, {1, 0}},      (fm::Rewards(2) << 2.0, 3.0).finished()},
    };

    const MOVE::Results solutions{std::make_tuple(fm::PartialAction{{0, 1}, {0, 0}}, (fm::Rewards(2) << 9.0, 1.0).finished()),
                                  std::make_tuple(fm::PartialAction{{0, 1}, {0, 1}}, (fm::Rewards(2) << 6.0, 2.0).finished()),
                                  std::make_tuple(fm::PartialAction{{0, 1}, {1, 0}}, (fm::Rewards(2) << 7.0, 4.0).finished()),
                                  std::make_tuple(fm::PartialAction{{0, 1}, {1, 1}}, (fm::Rewards(2) << 2.0, 2.0).finished())};

    fm::Action a{2, 2};

    MOVE v(a);
    auto bestActions = v(rules);

    BOOST_CHECK_EQUAL(solutions.size(), bestActions.size());

    for (size_t i = 0; i < solutions.size(); ++i) {
        const auto & spa1 = std::get<0>(solutions[i]).first;
        const auto & spa2 = std::get<0>(solutions[i]).second;

        const auto & pa1 = std::get<0>(bestActions[i]).first;
        const auto & pa2 = std::get<0>(bestActions[i]).second;

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(spa1), std::end(spa1), std::begin(pa1), std::end(pa1));
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(spa2), std::end(spa2), std::begin(pa2), std::end(pa2));

        BOOST_CHECK_EQUAL(std::get<1>(solutions[i]), std::get<1>(bestActions[i]));
    }
}

BOOST_AUTO_TEST_CASE( simple_graph_2 ) {
    std::vector<fm::MOQFunctionRule> rules {
        // States, Actions,                     Value
        {    {},   {{0}, {0}},            (fm::Rewards(2) << 4.0, 0,0).finished()},
        {    {},   {{0, 1}, {1, 0}},      (fm::Rewards(2) << 2.0, 3.0).finished()},
    };

    MOVE::Results solutions{std::make_tuple(fm::PartialAction{{0}, {0}},       (fm::Rewards(2) << 4.0, 0.0).finished()),
                            std::make_tuple(fm::PartialAction{{0, 1}, {1, 0}}, (fm::Rewards(2) << 2.0, 3.0).finished())};

    fm::Action a{2, 2};

    MOVE v(a);
    auto bestActions = v(rules);

    BOOST_CHECK_EQUAL(solutions.size(), bestActions.size());

    for (size_t i = 0; i < solutions.size(); ++i) {
        const auto & spa1 = std::get<0>(solutions[i]).first;
        const auto & spa2 = std::get<0>(solutions[i]).second;

        const auto & pa1 = std::get<0>(bestActions[i]).first;
        const auto & pa2 = std::get<0>(bestActions[i]).second;

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(spa1), std::end(spa1), std::begin(pa1), std::end(pa1));
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(spa2), std::end(spa2), std::begin(pa2), std::end(pa2));

        BOOST_CHECK_EQUAL(std::get<1>(solutions[i]), std::get<1>(bestActions[i]));
    }
}
