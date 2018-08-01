#define BOOST_TEST_MODULE Global
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Utils/Prune.hpp>

BOOST_AUTO_TEST_CASE( vector_comparisons ) {
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
    AIToolbox::RandomEngine rand(AIToolbox::Impl::Seeder::getSeed());

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

BOOST_AUTO_TEST_CASE( dominationPrune ) {
    using namespace AIToolbox;

    std::vector<std::vector<Vector>> data {{
        (Vector(2) <<   7.5975 , -96.9025).finished(),
        (Vector(2) <<  -8.0775 ,  -8.0775).finished(),
        (Vector(2) <<   6.03   , -16.96).finished(),
        (Vector(2) <<   7.29576, -28.3518).finished(),
        (Vector(2) <<   4.01968,  -9.78738).finished(),
        (Vector(2) << -81.2275 , -81.2275).finished(),
        (Vector(2) << -96.9025 ,   7.5975).finished(),
        (Vector(2) << -82.795  ,  -1.285).finished(),
        (Vector(2) << -81.5292 , -12.6768).finished(),
        (Vector(2) << -84.8053 ,   5.88762).finished(),
        (Vector(2) <<  -1.285  , -82.795).finished(),
        (Vector(2) << -16.96   ,   6.03).finished(),
        (Vector(2) <<  -2.8525 ,  -2.8525).finished(),
        (Vector(2) <<  -1.58674, -14.2443).finished(),
        (Vector(2) <<  -4.86282,   4.32012).finished(),
        (Vector(2) <<   5.88762, -84.8053).finished(),
        (Vector(2) <<  -9.78738,   4.01968).finished(),
        (Vector(2) <<   4.32012,  -4.86282).finished(),
        (Vector(2) <<   5.58587, -16.2546).finished(),
        (Vector(2) <<   2.3098 ,   2.3098).finished(),
        (Vector(2) << -12.6768 , -81.5292).finished(),
        (Vector(2) << -28.3518 ,   7.29576).finished(),
        (Vector(2) << -14.2443 ,  -1.58674).finished(),
        (Vector(2) << -12.9786 , -12.9786).finished(),
        (Vector(2) << -16.2546 ,   5.58587).finished(),
    }, {
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) << -100.000 ,  10.0000).finished(),
        (Vector(2) <<  10.0000 , -100.000).finished(),
    }, {
        // Test duplicates
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
    }};

    std::vector<std::vector<Vector>> solutions {{
        (Vector(2) <<   7.5975 , -96.9025).finished(),
        (Vector(2) << -16.2546 ,   5.58587).finished(),
        (Vector(2) <<   6.03   , -16.96).finished(),
        (Vector(2) <<   7.29576, -28.3518).finished(),
        (Vector(2) << -28.3518 ,   7.29576).finished(),
        (Vector(2) <<   2.3098 ,   2.3098).finished(),
        (Vector(2) << -96.9025 ,   7.5975).finished(),
        (Vector(2) <<   5.58587, -16.2546).finished(),
        (Vector(2) <<   4.32012,  -4.86282).finished(),
        (Vector(2) <<  -4.86282,   4.32012).finished(),
        (Vector(2) <<  -16.96  ,   6.03).finished(),
    }, {
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
        (Vector(2) << -100.000 ,  10.0000).finished(),
        (Vector(2) <<  10.0000 , -100.000).finished(),
    }, {
        (Vector(2) <<  -1.0000 ,  -1.0000).finished(),
    }};

    for (size_t i = 0; i < data.size(); ++i) {
        auto & d = data[i];
        auto & s = solutions[i];

        auto test = extractDominated(2, std::begin(d), std::end(d));
        auto comparer = [](const auto & lhs, const auto & rhs) {
            return veccmp(lhs, rhs) < 0;
        };

        std::sort(std::begin(s), std::end(s), comparer);
        std::sort(std::begin(d), test, comparer);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(s), std::end(s),
                                      std::begin(d), test);
    }
}
