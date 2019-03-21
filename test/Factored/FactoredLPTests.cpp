#define BOOST_TEST_MODULE Factored_MDP_FactoredLP
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/LP.hpp>
//#include <AIToolbox/Factored/MDP/Algorithms/FactoredValueIteration.hpp>
#include <AIToolbox/Factored/MDP/Algorithms/Utils/FactoredLP.hpp>

namespace aif = AIToolbox::Factored;
namespace fm = AIToolbox::Factored::MDP;
using FLP = fm::FactoredLP;

BOOST_AUTO_TEST_CASE( test_1 ) {
    aif::State s{2,2,2};

    aif::FactoredVector C;
    aif::BasisFunction c1{{0,1}, {}};
    c1.values.resize(4);
    c1.values << 1.0, 3.0, 2.0, 4.0;

    aif::BasisFunction c2{{0,2}, {}};
    c2.values.resize(4);
    c2.values << 7.0, 9.0, 8.0, 10.0;

    C.bases.emplace_back(std::move(c1));
    C.bases.emplace_back(std::move(c2));

    aif::FactoredVector b;
    aif::BasisFunction b1{{1,2}, {}};
    b1.values.resize(4);
    b1.values << 7.0, 10.0, 6.0, 9.0;

    aif::BasisFunction b2{{0,2}, {}};
    b2.values.resize(4);
    b2.values << 10.0, 20.0, 13.0, 23.0;

    b.bases.emplace_back(std::move(b1));
    b.bases.emplace_back(std::move(b2));

    fm::FactoredLP l(s);

    const auto result = l(C, b);
    const std::vector<double> solution{3.0, 2.0};

    BOOST_CHECK(result);
    BOOST_CHECK_EQUAL(result->size(), solution.size());

    // So here the results are not actually perfect (damn you floating point
    // errors!) and at the same time our default checking functions do not help
    // us since they'd like the results to be about 1000x more precise with
    // this numbers to accept the relative error w.r.t. the solution.
    //
    // So we "cheat" and use a function that hopefully gives us the average
    // precision of LP solutions, so we can compare them and have working
    // tests.
    for (size_t i = 0; i < solution.size(); ++i) {
        BOOST_TEST_INFO("Element " << i);
        BOOST_TEST_INFO("Solution: " << solution[i] << "; Result: " << (*result)[i]);
        BOOST_CHECK(std::fabs(solution[i] - (*result)[i]) < AIToolbox::LP::getPrecision());
    }
}

BOOST_AUTO_TEST_CASE( test_2 ) {
    aif::State s{2,2,2};

    aif::FactoredVector C;
    aif::BasisFunction c1{{0,1}, {}};
    c1.values.resize(4);
    c1.values << 10.0, 2.0, 5.0, 7.5;

    aif::BasisFunction c2{{0,2}, {}};
    c2.values.resize(4);
    c2.values << 4.5, 6.0, 2.0, 3.5;

    C.bases.emplace_back(std::move(c1));
    C.bases.emplace_back(std::move(c2));

    aif::FactoredVector b;
    aif::BasisFunction b1{{1,2}, {}};
    b1.values.resize(4);
    b1.values << 26.5, 21.75, 19.0, 14.25;

    aif::BasisFunction b2{{0,1}, {}};
    b2.values.resize(4);
    b2.values << 32.0, 0.5, 14.25, 30.0;

    b.bases.emplace_back(std::move(b1));
    b.bases.emplace_back(std::move(b2));

    fm::FactoredLP l(s);

    const auto result = l(C, b);
    const std::vector<double> solution{4.5, 3.0};

    BOOST_CHECK(result);
    BOOST_CHECK_EQUAL(result->size(), solution.size());

    // So here the results are not actually perfect (damn you floating point
    // errors!) and at the same time our default checking functions do not help
    // us since they'd like the results to be about 1000x more precise with
    // this numbers to accept the relative error w.r.t. the solution.
    //
    // So we "cheat" and use a function that hopefully gives us the average
    // precision of LP solutions, so we can compare them and have working
    // tests.
    for (size_t i = 0; i < solution.size(); ++i) {
        BOOST_TEST_INFO("Element " << i);
        BOOST_TEST_INFO("Solution: " << solution[i] << "; Result: " << (*result)[i]);
        BOOST_CHECK(std::fabs(solution[i] - (*result)[i]) < AIToolbox::LP::getPrecision());
    }
}

BOOST_AUTO_TEST_CASE( test_constant_basis ) {
    aif::State s{2,2,2};

    aif::FactoredVector C;
    aif::BasisFunction c1{{0,1}, {}};
    c1.values.resize(4);
    c1.values << 1.0, 3.0, 2.0, 4.0;

    aif::BasisFunction c2{{0,2}, {}};
    c2.values.resize(4);
    c2.values << 7.0, 9.0, 8.0, 10.0;

    C.bases.emplace_back(std::move(c1));
    C.bases.emplace_back(std::move(c2));

    aif::FactoredVector b;
    aif::BasisFunction b1{{1,2}, {}};
    b1.values.resize(4);
    b1.values << 6.0, 9.0, 5.0, 8.0;

    aif::BasisFunction b2{{0,2}, {}};
    b2.values.resize(4);
    b2.values << 9.0, 19.0, 12.0, 22.0;

    b.bases.emplace_back(std::move(b1));
    b.bases.emplace_back(std::move(b2));

    fm::FactoredLP l(s);

    const auto result = l(C, b, true);
    const std::vector<double> solution{3.0, 2.0, -2.0};

    BOOST_CHECK(result);
    BOOST_CHECK_EQUAL(result->size(), solution.size());

    // So here the results are not actually perfect (damn you floating point
    // errors!) and at the same time our default checking functions do not help
    // us since they'd like the results to be about 1000x more precise with
    // this numbers to accept the relative error w.r.t. the solution.
    //
    // So we "cheat" and use a function that hopefully gives us the average
    // precision of LP solutions, so we can compare them and have working
    // tests.
    for (size_t i = 0; i < solution.size(); ++i) {
        BOOST_TEST_INFO("Element " << i);
        BOOST_TEST_INFO("Solution: " << solution[i] << "; Result: " << (*result)[i]);
        BOOST_CHECK(std::fabs(solution[i] - (*result)[i]) < AIToolbox::LP::getPrecision());
    }
}
