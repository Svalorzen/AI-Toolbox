#define BOOST_TEST_MODULE Global
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>

BOOST_AUTO_TEST_CASE( veccmp ) {
    using namespace AIToolbox;

    using Test = std::tuple<std::vector<double>, std::vector<double>, int>;

    std::vector<Test> data {
        Test({1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, 0),
        Test({0.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, -1),
        Test({1.0, 1.0, 3.0}, {1.0, 2.0, 3.0}, -1),
        Test({1.0, 2.0, 2.0}, {1.0, 2.0, 3.0}, -1),
        Test({1.0, 2.0, 3.0}, {0.0, 2.0, 3.0}, 1),
        Test({1.0, 2.0, 3.0}, {1.0, 1.0, 3.0}, 1),
        Test({1.0, 2.0, 3.0}, {1.0, 2.0, 2.0}, 1),
    };

    for (const auto & test : data) {
        Vector lhs = Vector::Map(std::get<0>(test).data(), 3);
        Vector rhs = Vector::Map(std::get<1>(test).data(), 3);

        BOOST_CHECK_EQUAL(AIToolbox::veccmp(lhs, rhs), std::get<2>(test));
    }
}

BOOST_AUTO_TEST_CASE( probGeneration ) {
    std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());

    for (size_t i = 0; i < 100; ++i) {
        const auto v = AIToolbox::makeRandomProbability(i+1, rand);
        for (size_t j = 0; j < i+1; ++j)
            BOOST_CHECK(0.0 <= v[j] && v[j] <= 1.0);
        BOOST_CHECK(AIToolbox::checkEqualSmall(v.sum(), 1.0));
    }
}

BOOST_AUTO_TEST_CASE( probProjection ) {
    using namespace AIToolbox;

    std::vector<Vector> data {
        (Vector(3) << 1.0, 2.0, 3.0).finished(),
        (Vector(3) << 0.4, 0.6, 0.1).finished(),
        (Vector(3) << -1.0, 0.6, 0.6).finished(),
        (Vector(3) << -4.0, -7.0, -1.0).finished(),
        (Vector(3) << 0.3, -7.0, 0.2).finished(),
    };

    std::vector<Vector> solutions {
        (Vector(3) << 1.0/6.0, 2.0/6.0, 3.0/6.0).finished(),
        (Vector(3) << 0.4/1.1, 0.6/1.1, 0.1/1.1).finished(),
        (Vector(3) << 0.0, 0.6/1.2, 0.6/1.2).finished(),
        (Vector(3) << 1.0/3.0, 1.0/3.0, 1.0/3.0).finished(),
        (Vector(3) << 0.55, 0.0, 0.45).finished(),
    };

    for (size_t i = 0; i < data.size(); ++i) {
        BOOST_CHECK_EQUAL(AIToolbox::veccmp(projectToProbability(data[i]), solutions[i]), 0);
    }
}
