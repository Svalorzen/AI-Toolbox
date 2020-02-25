#define BOOST_TEST_MODULE UtilsPrune
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/Prune.hpp>

BOOST_AUTO_TEST_CASE( dominationPrune ) {
    using namespace AIToolbox;

    std::vector<std::vector<Hyperplane>> data {{
        (Hyperplane(2) <<   7.5975 , -96.9025).finished(),
        (Hyperplane(2) <<  -8.0775 ,  -8.0775).finished(),
        (Hyperplane(2) <<   6.03   , -16.96).finished(),
        (Hyperplane(2) <<   7.29576, -28.3518).finished(),
        (Hyperplane(2) <<   4.01968,  -9.78738).finished(),
        (Hyperplane(2) << -81.2275 , -81.2275).finished(),
        (Hyperplane(2) << -96.9025 ,   7.5975).finished(),
        (Hyperplane(2) << -82.795  ,  -1.285).finished(),
        (Hyperplane(2) << -81.5292 , -12.6768).finished(),
        (Hyperplane(2) << -84.8053 ,   5.88762).finished(),
        (Hyperplane(2) <<  -1.285  , -82.795).finished(),
        (Hyperplane(2) << -16.96   ,   6.03).finished(),
        (Hyperplane(2) <<  -2.8525 ,  -2.8525).finished(),
        (Hyperplane(2) <<  -1.58674, -14.2443).finished(),
        (Hyperplane(2) <<  -4.86282,   4.32012).finished(),
        (Hyperplane(2) <<   5.88762, -84.8053).finished(),
        (Hyperplane(2) <<  -9.78738,   4.01968).finished(),
        (Hyperplane(2) <<   4.32012,  -4.86282).finished(),
        (Hyperplane(2) <<   5.58587, -16.2546).finished(),
        (Hyperplane(2) <<   2.3098 ,   2.3098).finished(),
        (Hyperplane(2) << -12.6768 , -81.5292).finished(),
        (Hyperplane(2) << -28.3518 ,   7.29576).finished(),
        (Hyperplane(2) << -14.2443 ,  -1.58674).finished(),
        (Hyperplane(2) << -12.9786 , -12.9786).finished(),
        (Hyperplane(2) << -16.2546 ,   5.58587).finished(),
    }, {
        (Hyperplane(2) <<  -1.0000 ,  -1.0000).finished(),
        (Hyperplane(2) << -100.000 ,  10.0000).finished(),
        (Hyperplane(2) <<  10.0000 , -100.000).finished(),
    }, {
        // Test duplicates
        (Hyperplane(2) <<  -1.0000 ,  -1.0000).finished(),
        (Hyperplane(2) <<  -1.0000 ,  -1.0000).finished(),
        (Hyperplane(2) <<  -1.0000 ,  -1.0000).finished(),
        (Hyperplane(2) <<  -1.0000 ,  -1.0000).finished(),
        (Hyperplane(2) <<  -1.0000 ,  -1.0000).finished(),
    }};

    std::vector<std::vector<Hyperplane>> solutions {{
        (Hyperplane(2) <<   7.5975 , -96.9025).finished(),
        (Hyperplane(2) << -16.2546 ,   5.58587).finished(),
        (Hyperplane(2) <<   6.03   , -16.96).finished(),
        (Hyperplane(2) <<   7.29576, -28.3518).finished(),
        (Hyperplane(2) << -28.3518 ,   7.29576).finished(),
        (Hyperplane(2) <<   2.3098 ,   2.3098).finished(),
        (Hyperplane(2) << -96.9025 ,   7.5975).finished(),
        (Hyperplane(2) <<   5.58587, -16.2546).finished(),
        (Hyperplane(2) <<   4.32012,  -4.86282).finished(),
        (Hyperplane(2) <<  -4.86282,   4.32012).finished(),
        (Hyperplane(2) <<  -16.96  ,   6.03).finished(),
    }, {
        (Hyperplane(2) <<  -1.0000 ,  -1.0000).finished(),
        (Hyperplane(2) << -100.000 ,  10.0000).finished(),
        (Hyperplane(2) <<  10.0000 , -100.000).finished(),
    }, {
        (Hyperplane(2) <<  -1.0000 ,  -1.0000).finished(),
    }};

    for (size_t i = 0; i < data.size(); ++i) {
        auto & d = data[i];
        auto & s = solutions[i];

        auto test = extractDominated(std::begin(d), std::end(d));
        auto comparer = [](const auto & lhs, const auto & rhs) {
            return veccmp(lhs, rhs) < 0;
        };

        std::sort(std::begin(s), std::end(s), comparer);
        std::sort(std::begin(d), test, comparer);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(s), std::end(s),
                                      std::begin(d), test);
    }
}

BOOST_AUTO_TEST_CASE( dominationIncrementalPrune ) {
    using namespace AIToolbox;

    std::vector<Hyperplane> startSet {
        (Hyperplane(2) <<  10, -10).finished(),
        (Hyperplane(2) << -10,  10).finished(),
        (Hyperplane(2) <<   0,   0).finished(),
    };

    std::vector<std::vector<Hyperplane>> addSets {
        // Nothing to add
        {
        },
        // All new stuff is dominated
        {
            (Hyperplane(2) <<  9, -11).finished(),
            (Hyperplane(2) <<  -11, 9).finished(),
            (Hyperplane(2) <<  -1, -1).finished(),
        },
        {
            (Hyperplane(2) <<  0, -1).finished(),
        },
        // Some gets in
        {
            (Hyperplane(2) <<  15, -15).finished(), // in
            (Hyperplane(2) <<  -11, 8).finished(),
            (Hyperplane(2) <<  5, -20).finished(),
            (Hyperplane(2) <<  1, -1).finished(),   // in
        },
        // Some gets in and dominates new adds
        {
            (Hyperplane(2) <<  15, -15).finished(),
            (Hyperplane(2) <<  14, -16).finished(),
            (Hyperplane(2) <<  20, -14).finished(), // in
        },
        // Dominates some of old
        {
            (Hyperplane(2) <<  15, -15).finished(), // in
            (Hyperplane(2) <<  0, 1).finished(),    // repl
            (Hyperplane(2) <<  -11, 9).finished(),
        },
        {
            (Hyperplane(2) <<  10, 0).finished(),   // repl
            (Hyperplane(2) <<  -11, 9).finished(),
            (Hyperplane(2) <<  -15, 15).finished(),  // in
        },
        // Dominates everything
        {
            (Hyperplane(2) <<  1, 1).finished(),
            (Hyperplane(2) <<  2, 2).finished(),
            (Hyperplane(2) <<  5, 15).finished(),
            (Hyperplane(2) <<  100, 100).finished(),
        },
        {
            (Hyperplane(2) <<  100, 100).finished(),
            (Hyperplane(2) <<  5, 15).finished(),
            (Hyperplane(2) <<  2, 2).finished(),
            (Hyperplane(2) <<  1, 1).finished(),
        },
    };

    std::vector<std::vector<Hyperplane>> solutions {
        startSet, startSet, startSet,
        {
            (Hyperplane(2) <<  10, -10).finished(),
            (Hyperplane(2) << -10,  10).finished(),
            (Hyperplane(2) <<   0,   0).finished(),
            (Hyperplane(2) <<  15, -15).finished(), // in
            (Hyperplane(2) <<  1, -1).finished(),   // in
        },
        {
            (Hyperplane(2) <<  10, -10).finished(),
            (Hyperplane(2) << -10,  10).finished(),
            (Hyperplane(2) <<   0,   0).finished(),
            (Hyperplane(2) <<  20, -14).finished(), // in
        },
        {
            (Hyperplane(2) <<  10, -10).finished(),
            (Hyperplane(2) << -10,  10).finished(),
            (Hyperplane(2) <<   0,   1).finished(), // repl
            (Hyperplane(2) <<  15, -15).finished(), // in
        },
        {
            (Hyperplane(2) <<  10,   0).finished(), // repl
            (Hyperplane(2) << -10,  10).finished(),
            (Hyperplane(2) <<  -15, 15).finished(),  // in
        },
        {
            (Hyperplane(2) <<  100, 100).finished(),
        },
        {
            (Hyperplane(2) <<  100, 100).finished(),
        }
    };

    for (size_t i = 0; i < addSets.size(); ++i) {
        auto start = startSet; // Copy since we are going to modify it
        size_t startSize = start.size();
        start.insert(std::end(start), std::begin(addSets[i]), std::end(addSets[i]));

        auto & s = solutions[i];

        auto test = extractDominatedIncremental(std::begin(start), std::begin(start) + startSize, std::end(start));

        auto comparer = [](const auto & lhs, const auto & rhs) {
            return veccmp(lhs, rhs) < 0;
        };

        std::sort(std::begin(s), std::end(s), comparer);
        std::sort(std::begin(start), test, comparer);
        BOOST_TEST_INFO(i);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(s), std::end(s),
                                      std::begin(start), test);
    }
}
