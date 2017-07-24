#define BOOST_TEST_MODULE FactoredMDP_FactoredLP
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/LP.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/Utils/FactoredLP.hpp>

namespace fm = AIToolbox::FactoredMDP;
using FLP = fm::FactoredLP;

BOOST_AUTO_TEST_CASE( test_1 ) {
    fm::State s{2,2,2};

    std::vector<fm::ValueFunctionRule> r1 {
        {{{0, 1}, {0, 0}}, 1.0},
        {{{0, 1}, {0, 1}}, 2.0},
        {{{0, 1}, {1, 0}}, 3.0},
        {{{0, 1}, {1, 1}}, 4.0},
    };

    std::vector<fm::ValueFunctionRule> r2 {
        {{{0, 2}, {0, 0}}, 7.0},
        {{{0, 2}, {0, 1}}, 8.0},
        {{{0, 2}, {1, 0}}, 9.0},
        {{{0, 2}, {1, 1}}, 10.0},
    };

    FLP::FactoredFunction C(3);
    C.getFactor({0, 1})->getData() = r1;
    C.getFactor({0, 2})->getData() = r2;

    std::vector<fm::ValueFunctionRule> r3 {
        {{{1, 2}, {0, 0}}, 7.0},
        {{{1, 2}, {0, 1}}, 6.0},
        {{{1, 2}, {1, 0}}, 10.0},
        {{{1, 2}, {1, 1}}, 9.0},
    };

    std::vector<fm::ValueFunctionRule> r4 {
        {{{0, 2}, {0, 0}}, 10.0},
        {{{0, 2}, {0, 1}}, 13.0},
        {{{0, 2}, {1, 0}}, 20.0},
        {{{0, 2}, {1, 1}}, 23.0},
    };

    FLP::FactoredFunction b(3);
    b.getFactor({1, 2})->getData() = r3;
    b.getFactor({0, 2})->getData() = r4;

    fm::FactoredLP l(s);
    const auto result = l(C, b);
    const std::vector<double> solution{3.0, 2.0};

    BOOST_CHECK(result);
    BOOST_CHECK_EQUAL(result->size(), 2);

    // So here the results are not actually perfect (damn you floating point
    // errors!) and at the same time our default checking functions do not help
    // us since they'd like the results to be about 1000x more precise with
    // this numbers to accept the relative error w.r.t. the solution.
    //
    // So we "cheat" and use a function that hopefully gives us the average
    // precision of LP solutions, so we can compare them and have working
    // tests.
    for (size_t i = 0; i < solution.size(); ++i)
        BOOST_CHECK(std::fabs(solution[i] - (*result)[i]) < AIToolbox::LP::getPrecision());
}

BOOST_AUTO_TEST_CASE( test_2 ) {
    fm::State s{2,2,2};

    std::vector<fm::ValueFunctionRule> r1 {
        {{{0, 1}, {0, 0}}, 10.0},
        {{{0, 1}, {0, 1}}, 5.0},
        {{{0, 1}, {1, 0}}, 2.0},
        {{{0, 1}, {1, 1}}, 7.5},
    };

    std::vector<fm::ValueFunctionRule> r2 {
        {{{0, 2}, {0, 0}}, 4.5},
        {{{0, 2}, {0, 1}}, 2.0},
        {{{0, 2}, {1, 0}}, 6.0},
        {{{0, 2}, {1, 1}}, 3.5},
    };

    FLP::FactoredFunction C(3);
    C.getFactor({0, 1})->getData() = r1;
    C.getFactor({0, 2})->getData() = r2;

    std::vector<fm::ValueFunctionRule> r3 {
        {{{1, 2}, {0, 0}}, 26.5},
        {{{1, 2}, {0, 1}}, 19.0},
        {{{1, 2}, {1, 0}}, 21.75},
        {{{1, 2}, {1, 1}}, 14.25},
    };

    std::vector<fm::ValueFunctionRule> r4 {
        {{{0, 1}, {0, 0}}, 32.0},
        {{{0, 1}, {0, 1}}, 14.25},
        {{{0, 1}, {1, 0}}, 0.5},
        {{{0, 1}, {1, 1}}, 30.0},
    };

    FLP::FactoredFunction b(3);
    b.getFactor({1, 2})->getData() = r3;
    b.getFactor({0, 2})->getData() = r4;

    fm::FactoredLP l(s);

    const auto result = l(C, b);
    const std::vector<double> solution{4.5, 3.0};

    BOOST_CHECK(result);
    BOOST_CHECK_EQUAL(result->size(), 2);

    // So here the results are not actually perfect (damn you floating point
    // errors!) and at the same time our default checking functions do not help
    // us since they'd like the results to be about 1000x more precise with
    // this numbers to accept the relative error w.r.t. the solution.
    //
    // So we "cheat" and use a function that hopefully gives us the average
    // precision of LP solutions, so we can compare them and have working
    // tests.
    for (size_t i = 0; i < solution.size(); ++i)
        BOOST_CHECK(std::fabs(solution[i] - (*result)[i]) < AIToolbox::LP::getPrecision());
}
