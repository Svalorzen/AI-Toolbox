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

    MOVE::Results solutions{std::make_tuple(fm::Action{0, 0}, (fm::Rewards(2) << 9.0, 1.0).finished()),
                            std::make_tuple(fm::Action{0, 1}, (fm::Rewards(2) << 6.0, 2.0).finished()),
                            std::make_tuple(fm::Action{1, 0}, (fm::Rewards(2) << 7.0, 4.0).finished()),
                            std::make_tuple(fm::Action{1, 1}, (fm::Rewards(2) << 2.0, 2.0).finished())};

    fm::Action a{2, 2};

    MOVE v(a);
    auto bestActions = v(rules);

    for (const auto & a : bestActions) {
        std::cout << "[";
        for (const auto & aa : std::get<0>(a))
            std::cout << aa << ", ";
        std::cout << "] ==> [";
        std::cout << std::get<1>(a).transpose() << "]\n";
    }

    BOOST_CHECK_EQUAL(solutions.size(), bestActions.size());

    // for (size_t i =
    // BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(bestActions),  std::end(bestActions),
    //                               std::begin(solutions),    std::end(solutions));
}

