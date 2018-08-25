#define BOOST_TEST_MODULE UtilsProbability
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

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

BOOST_AUTO_TEST_CASE( vose_alias_sampling ) {
    AIToolbox::RandomEngine rand(AIToolbox::Impl::Seeder::getSeed());

    AIToolbox::ProbabilityVector p(7);
    p << 1.0/8, 1.0/5, 1.0/10, 1.0/4, 1.0/10, 1.0/10, 1.0/8;

    AIToolbox::VoseAliasSampler vose(p);

    constexpr size_t trials = 100'000;
    std::vector<size_t> counters(p.size());
    for (size_t i = 0; i < trials; ++i)
        ++counters[vose.sampleProbability(rand)];

    constexpr double percentageErrorAllowed = 0.05;

    for (size_t i = 0; i < counters.size(); ++i) {
        const auto exactAmount = p[i] * trials;
        BOOST_CHECK(std::abs(counters[i] - exactAmount) < percentageErrorAllowed * exactAmount);
    }
}
