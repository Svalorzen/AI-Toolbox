#define BOOST_TEST_MODULE FactoredMDP_Utils
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <AIToolbox/FactoredMDP/Utils.hpp>

namespace fm = AIToolbox::FactoredMDP;

BOOST_AUTO_TEST_CASE( partial_factor_merge ) {
    fm::PartialFactors lhs = {{0, 3, 5, 6}, {0, 3, 5, 6}};
    fm::PartialFactors rhs = {{1, 2, 4, 7}, {1, 2, 4, 7}};

    fm::PartialFactors solution = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    auto result1 = fm::merge(lhs, rhs);

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution.first), std::end(solution.first),
                                  std::begin(result1.first), std::end(result1.first));

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution.second), std::end(solution.second),
                                  std::begin(result1.second), std::end(result1.second));

    auto result2 = fm::merge(rhs, lhs);

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution.first), std::end(solution.first),
                                  std::begin(result2.first), std::end(result2.first));

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution.second), std::end(solution.second),
                                  std::begin(result2.second), std::end(result2.second));
}

BOOST_AUTO_TEST_CASE( partial_factor_enumerator_no_skip ) {
    fm::Factors f{1,2,3,4,5};
    fm::PartialFactorsEnumerator enumerator(f, {0, 2, 3});

    std::vector<fm::PartialAction> solution{
        {{0, 2, 3}, {0, 0, 0}},
        {{0, 2, 3}, {0, 1, 0}},
        {{0, 2, 3}, {0, 2, 0}},
        {{0, 2, 3}, {0, 0, 1}},
        {{0, 2, 3}, {0, 1, 1}},
        {{0, 2, 3}, {0, 2, 1}},
        {{0, 2, 3}, {0, 0, 2}},
        {{0, 2, 3}, {0, 1, 2}},
        {{0, 2, 3}, {0, 2, 2}},
        {{0, 2, 3}, {0, 0, 3}},
        {{0, 2, 3}, {0, 1, 3}},
        {{0, 2, 3}, {0, 2, 3}},
    };

    size_t counter = 0;
    while (enumerator.isValid()) {
        const auto & val = *enumerator;
        const auto & sol = solution[counter];
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.first), std::end(val.first),
                                      std::begin(sol.first), std::end(sol.first));
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.second), std::end(val.second),
                                      std::begin(sol.second), std::end(sol.second));
        enumerator.advance();
        ++counter;
    }
}

BOOST_AUTO_TEST_CASE( partial_factor_enumerator_skip ) {
    fm::Factors f{1,2,3,4,5};
    fm::PartialFactorsEnumerator enumerator(f, {1, 3, 4}, 3);
    auto agentToSkip = enumerator.getFactorToSkipId();

    std::vector<fm::PartialAction> solution{
        {{1, 3, 4}, {0, 0, 0}},
        {{1, 3, 4}, {1, 1, 0}},
        {{1, 3, 4}, {0, 2, 1}},
        {{1, 3, 4}, {1, 3, 1}},
        {{1, 3, 4}, {0, 4, 2}},
        {{1, 3, 4}, {1, 5, 2}},
        {{1, 3, 4}, {0, 6, 3}},
        {{1, 3, 4}, {1, 7, 3}},
        {{1, 3, 4}, {0, 8, 4}},
        {{1, 3, 4}, {1, 9, 4}},
    };

    size_t counter = 0;
    while (enumerator.isValid()) {
        auto val = *enumerator;
        const auto & sol = solution[counter];
        // Modify value
        val.second[agentToSkip] = counter;

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.first), std::end(val.first),
                                      std::begin(sol.first), std::end(sol.first));
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.second), std::end(val.second),
                                      std::begin(sol.second), std::end(sol.second));

        enumerator.advance();
        ++counter;
    }
}

BOOST_AUTO_TEST_CASE( partial_factor_enumerator_skip_only_factor ) {
    fm::Factors f{1,2,3,4,5};
    fm::PartialFactorsEnumerator enumerator(f, {0}, 0);

    auto agentToSkip = enumerator.getFactorToSkipId();

    std::vector<fm::PartialAction> solution{
        {{0}, {0}},
    };

    size_t counter = 0;
    while (enumerator.isValid()) {
        auto val = *enumerator;
        const auto & sol = solution[counter];
        // Modify value
        val.second[agentToSkip] = counter;

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.first), std::end(val.first),
                                      std::begin(sol.first), std::end(sol.first));
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.second), std::end(val.second),
                                      std::begin(sol.second), std::end(sol.second));

        enumerator.advance();
        ++counter;
    }
}
