#define BOOST_TEST_MODULE FactoredMDP_Utils
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/FactoredMDP/Utils.hpp>

#include <algorithm>

namespace fm = AIToolbox::FactoredMDP;

BOOST_AUTO_TEST_CASE( to_index_full_factors ) {
    fm::Factors state = {3,2,5};

    std::vector<size_t> solution;
    solution.resize(3*2*5);

    std::iota(std::begin(solution), std::end(solution), 0);

    std::vector<size_t> results;
    results.reserve(3*2*5);

    fm::Factors f = {0,0,0};
    for (size_t i = 0; i < 3*2*5; ++i) {
        results.push_back(fm::toIndex(state, f));

        // Get next factor.
        ++f[0];
        // Handle carry-over. We stop at size()-1 since the last iteration does
        // not need to go back to 0,0,0 and handling that is annoying.
        size_t j = 0;
        while (j < state.size()-1 && f[j] == state[j]) {
            f[j] = 0;
            ++f[++j];
        }
    }

    std::sort(std::begin(results), std::end(results));

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution), std::end(solution),
                                  std::begin(results), std::end(results));
}

BOOST_AUTO_TEST_CASE( to_index_full_partial_factors ) {
    fm::Factors state = {3,2,5};

    fm::PartialFactorsEnumerator enumerator(state, {0, 2});

    while (enumerator.isValid()) {
        auto val = *enumerator;
        // Copy the PartialFactors to Factors so we can use that logic to test
        // this one.
        auto fullFactor = fm::Factors{0,0,0};
        for (size_t i = 0; i < val.first.size(); ++i)
            fullFactor[val.first[i]] = val.second[i];

        BOOST_CHECK_EQUAL(fm::toIndex(state, val), fm::toIndex(state, fullFactor));

        enumerator.advance();
    }
}

BOOST_AUTO_TEST_CASE( to_index_partial_ids_factors ) {
    fm::Factors state = {3,2,5,4};
    std::vector<size_t> unusedids = {0, 2};
    std::vector<size_t> ids = {1, 3};

    std::vector<size_t> solution;
    solution.resize(2*4);

    std::iota(std::begin(solution), std::end(solution), 0);

    std::vector<size_t> results;
    results.reserve(2*4);

    // We iterate over the useless factors to check they are not being used.
    fm::Factors f = {0,0,0,0};
    for (size_t k = 0; k < 3*5; ++k) {
        // Reset results
        results.clear();
        // Reset parts of factor we care about
        for (auto id : ids)
            f[id] = 0;
        // Start testing
        for (size_t i = 0; i < 2*4; ++i) {
            results.push_back(fm::toIndexPartial(ids, state, f));

            // Get next factor.
            ++f[ids[0]];
            // Handle carry-over.
            size_t j = 0;
            while (f[ids[j]] == state[ids[j]]) {
                f[ids[j]] = 0;
                if (j == ids.size() - 1) break;
                ++f[ids[++j]];
            }
        }

        std::sort(std::begin(results), std::end(results));

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution), std::end(solution),
                                      std::begin(results), std::end(results));

        // Modify an unused ids to check it does not matter.
        ++f[unusedids[0]];
        // Handle carry-over.
        size_t j = 0;
        while (f[unusedids[j]] == state[unusedids[j]]) {
            f[unusedids[j]] = 0;
            if (j == unusedids.size() - 1) break;
            ++f[unusedids[++j]];
        }
    }
}

BOOST_AUTO_TEST_CASE( to_index_partial_partial_factor ) {
    fm::Factors state = {3,2,5,4};
    fm::PartialFactorsEnumerator enumerator(state, {0, 2});

    while (enumerator.isValid()) {
        auto val = *enumerator;
        // We can use toFactors here since we don't care about the value of
        // unneeded factors.
        auto fullFactor = fm::toFactors(state.size(), val);

        BOOST_CHECK_EQUAL(fm::toIndexPartial(state, val), fm::toIndexPartial(val.first, state, fullFactor));

        enumerator.advance();
    }
}

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
