#define BOOST_TEST_MODULE Factored_Utils
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

#include <algorithm>

namespace aif = AIToolbox::Factored;

BOOST_AUTO_TEST_CASE( match ) {
    std::vector<aif::PartialFactors> lhs = {
        {{2,3,5}, {1,2,3}},
        {{2,3,5}, {2,3,4}},
        {{1,2,3,5,6}, {1,2,3,4,5}},
        {{1,2,3}, {1,2,3}},
        {{1,4,6}, {1,2,3}},
        {{4,5,6}, {2,2,3}},
        {{3},     {2}},

        {{1,2,3}, {1,2,3}},
        {{1,2,3}, {2,2,3}},
        {{1,2,3}, {1,2,3}},
        {{2},     {3}},
    };
    std::vector<aif::PartialFactors> rhs = {
        {{2,3,5}, {1,2,3}},
        {{1,2,3,5,6}, {1,2,3,4,5}},
        {{2,3,5}, {2,3,4}},
        {{4,5,6}, {1,2,3}},
        {{4,5,6}, {2,2,3}},
        {{1,4,6}, {1,2,3}},
        {{2},     {3}},

        {{1,2,3}, {2,2,3}},
        {{1,2,3}, {1,2,3}},
        {{2},     {3}},
        {{1,2,3}, {1,2,3}},
    };
    std::vector<bool> solutions = {
        true,
        true,
        true,
        true,
        true,
        true,
        true,

        false,
        false,
        false,
        false,
    };

    for (size_t i = 0; i < solutions.size(); ++i) {
        BOOST_TEST_INFO(i);
        BOOST_CHECK_EQUAL(solutions[i], aif::match(lhs[i], rhs[i]));
    }
}

BOOST_AUTO_TEST_CASE( to_index_full_factors ) {
    aif::Factors state = {3,2,5};

    std::vector<size_t> solution;
    solution.resize(3*2*5);

    std::iota(std::begin(solution), std::end(solution), 0);

    std::vector<size_t> results;
    results.reserve(3*2*5);

    aif::Factors f = {0,0,0};
    for (size_t i = 0; i < 3*2*5; ++i) {
        results.push_back(aif::toIndex(state, f));

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
    aif::Factors state = {3,2,5};

    aif::PartialFactorsEnumerator enumerator(state, {0, 2});

    while (enumerator.isValid()) {
        auto val = *enumerator;
        // Copy the PartialFactors to Factors so we can use that logic to test
        // this one.
        auto fullFactor = aif::Factors{0,0,0};
        for (size_t i = 0; i < val.first.size(); ++i)
            fullFactor[val.first[i]] = val.second[i];

        BOOST_CHECK_EQUAL(aif::toIndex(state, val), aif::toIndex(state, fullFactor));

        enumerator.advance();
    }
}

BOOST_AUTO_TEST_CASE( to_index_partial_ids_factors ) {
    aif::Factors state = {3,2,5,4};
    std::vector<size_t> unusedids = {0, 2};
    std::vector<size_t> ids = {1, 3};

    std::vector<size_t> solution;
    solution.resize(2*4);

    std::iota(std::begin(solution), std::end(solution), 0);

    std::vector<size_t> results;
    results.reserve(2*4);

    // We iterate over the useless factors to check they are not being used.
    aif::Factors f = {0,0,0,0};
    for (size_t k = 0; k < 3*5; ++k) {
        // Reset results
        results.clear();
        // Reset parts of factor we care about
        for (auto id : ids)
            f[id] = 0;
        // Start testing
        for (size_t i = 0; i < 2*4; ++i) {
            results.push_back(aif::toIndexPartial(ids, state, f));

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
    aif::Factors state = {3,2,5,4};
    aif::PartialFactorsEnumerator enumerator(state, {0, 2});

    while (enumerator.isValid()) {
        auto val = *enumerator;
        // We can use toFactors here since we don't care about the value of
        // unneeded factors.
        auto fullFactor = aif::toFactors(state.size(), val);

        BOOST_CHECK_EQUAL(aif::toIndexPartial(state, val), aif::toIndexPartial(val.first, state, fullFactor));

        enumerator.advance();
    }
}

BOOST_AUTO_TEST_CASE( partial_keys_merge ) {
    std::vector<aif::PartialKeys> kl = {
        {},
        {},
        {0, 3, 4},
        {0, 3, 4},
        {1, 2, 3, 4, 5},
        {144, 200}
    };

    std::vector<aif::PartialKeys> kr = {
        {},
        {1, 3, 4},
        {1, 2, 5},
        {1, 3, 5},
        {1, 2, 3, 4, 5},
        {144, 198, 199}
    };

    std::vector<aif::PartialKeys> sol = {
        {},
        {1, 3, 4},
        {0, 1, 2, 3, 4, 5},
        {0, 1, 3, 4, 5},
        {1, 2, 3, 4, 5},
        {144, 198, 199, 200}
    };

    for (size_t i = 0; i < kl.size(); ++i) {
        const auto result = aif::merge(kl[i], kr[i]);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(sol[i]), std::end(sol[i]),
                                      std::begin(result), std::end(result));
    }
}

BOOST_AUTO_TEST_CASE( partial_factor_merge ) {
    aif::PartialFactors lhs = {{0, 3, 5, 6}, {0, 3, 5, 6}};
    aif::PartialFactors rhs = {{1, 2, 4, 7}, {1, 2, 4, 7}};

    aif::PartialFactors solution = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    auto result1 = aif::merge(lhs, rhs);

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution.first), std::end(solution.first),
                                  std::begin(result1.first), std::end(result1.first));

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution.second), std::end(solution.second),
                                  std::begin(result1.second), std::end(result1.second));

    auto result2 = aif::merge(rhs, lhs);

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution.first), std::end(solution.first),
                                  std::begin(result2.first), std::end(result2.first));

    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(solution.second), std::end(solution.second),
                                  std::begin(result2.second), std::end(result2.second));
}

BOOST_AUTO_TEST_CASE( partial_factor_enumerator_no_skip ) {
    aif::Factors f{1,2,3,4,5};
    aif::PartialFactorsEnumerator enumerator(f, {0, 2, 3});

    std::vector<aif::PartialAction> solution{
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

    for (size_t counter = 0; enumerator.isValid(); enumerator.advance(), ++counter) {
        const auto & val = *enumerator;
        const auto & sol = solution[counter];
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.first), std::end(val.first),
                                      std::begin(sol.first), std::end(sol.first));
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.second), std::end(val.second),
                                      std::begin(sol.second), std::end(sol.second));
    }
}

BOOST_AUTO_TEST_CASE( partial_factor_enumerator_skip ) {
    aif::Factors f{1,2,3,4,5};
    aif::PartialFactorsEnumerator enumerator(f, {1, 3, 4}, 3);
    auto agentToSkip = enumerator.getFactorToSkipId();

    std::vector<aif::PartialAction> solution{
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

    for (size_t counter = 0; enumerator.isValid(); enumerator.advance(), ++counter) {
        auto val = *enumerator;
        const auto & sol = solution[counter];
        // Modify value
        val.second[agentToSkip] = counter;

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.first), std::end(val.first),
                                      std::begin(sol.first), std::end(sol.first));
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.second), std::end(val.second),
                                      std::begin(sol.second), std::end(sol.second));
    }
}

BOOST_AUTO_TEST_CASE( partial_factor_enumerator_skip_missing ) {
    aif::Factors f{1,2,3,4,5};
    aif::PartialFactorsEnumerator enumerator(f, {1, 4}, 3, true);
    auto agentToSkip = enumerator.getFactorToSkipId();

    std::vector<aif::PartialAction> solution{
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

    for (size_t counter = 0; enumerator.isValid(); enumerator.advance(), ++counter) {
        auto val = *enumerator;
        const auto & sol = solution[counter];
        // Modify value
        val.second[agentToSkip] = counter;

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.first), std::end(val.first),
                                      std::begin(sol.first), std::end(sol.first));
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.second), std::end(val.second),
                                      std::begin(sol.second), std::end(sol.second));
    }
}

BOOST_AUTO_TEST_CASE( partial_factor_enumerator_skip_only_factor ) {
    aif::Factors f{1,2,3,4,5};
    aif::PartialFactorsEnumerator enumerator(f, {0}, 0);

    auto agentToSkip = enumerator.getFactorToSkipId();

    std::vector<aif::PartialAction> solution{
        {{0}, {0}},
    };

    for (size_t counter = 0; enumerator.isValid(); enumerator.advance(), ++counter) {
        auto val = *enumerator;
        const auto & sol = solution[counter];
        // Modify value
        val.second[agentToSkip] = counter;

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.first), std::end(val.first),
                                      std::begin(sol.first), std::end(sol.first));
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val.second), std::end(val.second),
                                      std::begin(sol.second), std::end(sol.second));
    }
}

BOOST_AUTO_TEST_CASE( partial_factor_enumerator_api_compatibility ) {
    aif::Factors f{1,2,3,4,5};
    aif::PartialFactorsEnumerator enumerator(f);

    for (size_t counter = 0; enumerator.isValid(); enumerator.advance(), ++counter) {
        const auto & val = enumerator->second;
        auto cmp   = aif::toFactors(f, counter);

        const auto cCmp  = aif::toIndex(f, *enumerator);


        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(val), std::end(val),
                                      std::begin(cmp), std::end(cmp));

        BOOST_CHECK_EQUAL(cCmp, counter);
    }
}
