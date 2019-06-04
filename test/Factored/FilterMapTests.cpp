#define BOOST_TEST_MODULE Factored_FilterMap
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Trie.hpp>
#include <AIToolbox/Factored/Utils/FasterTrie.hpp>
#include <AIToolbox/Factored/Utils/FilterMap.hpp>
#include <string>

BOOST_AUTO_TEST_CASE( construction ) {
    using namespace AIToolbox::Factored;
    Factors F{1,2,3};

    FilterMap<std::string, Trie> f(F);

    BOOST_CHECK(f.getF() == F);
    BOOST_CHECK_EQUAL(f.size(), 0);
}

BOOST_AUTO_TEST_CASE( filtering ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    FilterMap<std::string, Trie> f(F);

    f.emplace({{0,2},   {1,3}},     "1_3");
    f.emplace({{2},     {2}},       "__2");
    f.emplace({{1,2},   {0,0}},     "_00");
    f.emplace({{1},     {1}},       "_1_");
    f.emplace({{0},     {0}},       "0__");
    f.emplace({{1},     {2}},       "_2_");
    f.emplace({{1,2},   {0,1}},     "_01");
    f.emplace({{0},     {1}},       "1__");
    f.emplace({{0,1},   {0,0}},     "00_");
    f.emplace({{0,2},   {1,1}},     "1_1");
    f.emplace({{1,2},   {2,2}},     "_22");
    f.emplace({{0,1,2}, {1,1,1}},   "111");
    f.emplace({{1,2},   {2,0}},     "_20");
    f.emplace({{1,2},   {0,3}},     "_03");
    f.emplace({{0,2},   {1,2}},     "1_2");
    f.emplace({{0,2},   {1,0}},     "1_0");

    // This part will need to be upgraded to BOOST_DATA_TEST_CASE sooner or later.
    std::vector<Factors> filters{
        {0, 0, 0},
        {1, 2, 3},
        {0, 1, 2},
        {1, 0, 1},
        {0, 0, 3},
        {1, 1, 1}
    };
    std::vector<std::vector<std::string>> solutions{
        {"_00", "0__", "00_"},
        {"1_3", "_2_", "1__"},
        {"__2", "_1_", "0__"},
        {"_01", "1__", "1_1"},
        {"0__", "00_", "_03"},
        {"_1_", "1__", "1_1", "111"}
    };

    for (size_t i = 0; i < filters.size(); ++i) {
        auto filtered = f.filter(filters[i]);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(solutions[i]), std::end(solutions[i]));
    }
}

BOOST_AUTO_TEST_CASE( partial_filtering_factors ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    FilterMap<std::string, Trie> f(F);

    f.emplace({{0,2},   {1,3}},     "1_3");
    f.emplace({{2},     {2}},       "__2");
    f.emplace({{1,2},   {0,0}},     "_00");
    f.emplace({{1},     {1}},       "_1_");
    f.emplace({{0},     {0}},       "0__");
    f.emplace({{1},     {2}},       "_2_");
    f.emplace({{1,2},   {0,1}},     "_01");
    f.emplace({{0},     {1}},       "1__");
    f.emplace({{0,1},   {0,0}},     "00_");
    f.emplace({{0,2},   {1,1}},     "1_1");
    f.emplace({{1,2},   {2,2}},     "_22");
    f.emplace({{0,1,2}, {1,1,1}},   "111");
    f.emplace({{1,2},   {2,0}},     "_20");
    f.emplace({{1,2},   {0,3}},     "_03");
    f.emplace({{0,2},   {1,2}},     "1_2");
    f.emplace({{0,2},   {1,0}},     "1_0");

    // This part will need to be upgraded to BOOST_DATA_TEST_CASE sooner or later.
    std::vector<std::pair<Factors, size_t>> filters{
        {{0}, 2},       // All that end with 0
        {{1, 2}, 0},    // All that begin with 1,2
        {{1}, 1},       // All with 1 in the middle
        {{1, 0}, 0},    // All that begin with 1,0
        {{0, 1}, 1},    // All that end with 0,1
    };
    std::vector<std::vector<std::string>> solutions{
        {"_00", "_1_", "0__", "_2_", "1__", "00_", "_20", "1_0"},
        {"1_3", "__2", "_2_", "1__", "1_1", "_22", "_20", "1_2", "1_0"},
        {"1_3", "__2", "_1_", "0__", "1__", "1_1", "111", "1_2", "1_0"},
        {"1_3", "__2", "_00", "_01", "1__", "1_1", "_03", "1_2", "1_0"},
        {"0__", "_01", "1__", "00_", "1_1"}
    };

    for (size_t i = 0; i < filters.size(); ++i) {
        auto filtered = f.filter(filters[i].first, filters[i].second);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(solutions[i]), std::end(solutions[i]));
    }
}

BOOST_AUTO_TEST_CASE( partial_filtering_partial_factors ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    FilterMap<std::string, Trie> f(F);

    f.emplace({{0,2},   {1,3}},     "1_3");
    f.emplace({{2},     {2}},       "__2");
    f.emplace({{1,2},   {0,0}},     "_00");
    f.emplace({{1},     {1}},       "_1_");
    f.emplace({{0},     {0}},       "0__");
    f.emplace({{1},     {2}},       "_2_");
    f.emplace({{1,2},   {0,1}},     "_01");
    f.emplace({{0},     {1}},       "1__");
    f.emplace({{0,1},   {0,0}},     "00_");
    f.emplace({{0,2},   {1,1}},     "1_1");
    f.emplace({{1,2},   {2,2}},     "_22");
    f.emplace({{0,1,2}, {1,1,1}},   "111");
    f.emplace({{1,2},   {2,0}},     "_20");
    f.emplace({{1,2},   {0,3}},     "_03");
    f.emplace({{0,2},   {1,2}},     "1_2");
    f.emplace({{0,2},   {1,0}},     "1_0");

    // This part will need to be upgraded to BOOST_DATA_TEST_CASE sooner or later.
    std::vector<PartialFactors> filters{
        {{2},    {0}},       // All that end with 0
        {{0, 1}, {1, 2}},    // All that begin with 1,2
        {{1},    {1}},       // All with 1 in the middle
        {{0, 1}, {1, 0}},    // All that begin with 1,0
        {{1, 2}, {0, 1}},    // All that end with 0,1
    };
    std::vector<std::vector<std::string>> solutions{
        {"_00", "_1_", "0__", "_2_", "1__", "00_", "_20", "1_0"},
        {"1_3", "__2", "_2_", "1__", "1_1", "_22", "_20", "1_2", "1_0"},
        {"1_3", "__2", "_1_", "0__", "1__", "1_1", "111", "1_2", "1_0"},
        {"1_3", "__2", "_00", "_01", "1__", "1_1", "_03", "1_2", "1_0"},
        {"0__", "_01", "1__", "00_", "1_1"}
    };

    for (size_t i = 0; i < filters.size(); ++i) {
        auto filtered = f.filter(filters[i]);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(solutions[i]), std::end(solutions[i]));
    }
}

BOOST_AUTO_TEST_CASE( empty_filter ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    FilterMap<std::string, Trie> f(F);

    f.emplace({{0,2},   {1,3}},     "1_3");
    f.emplace({{2},     {2}},       "__2");
    f.emplace({{1,2},   {0,0}},     "_00");

    // This part will need to be upgraded to BOOST_DATA_TEST_CASE sooner or later.
    std::vector<Factors> filters{
        {0, 2, 3},
        {1, 2, 1},
        {1, 1, 0},
    };
    std::vector<std::string> solution;

    for (size_t i = 0; i < filters.size(); ++i) {
        auto filtered = f.filter(filters[i]);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(solution), std::end(solution));
    }
}

BOOST_AUTO_TEST_CASE( refine_partial_factors ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    FilterMap<std::string, Trie> f(F);

    f.emplace({{0,2},   {1,3}},     "1_3"); // 0
    f.emplace({{2},     {2}},       "__2"); // 1
    f.emplace({{1,2},   {0,0}},     "_00"); // 2
    f.emplace({{1},     {1}},       "_1_"); // 3
    f.emplace({{0},     {0}},       "0__"); // 4
    f.emplace({{1},     {2}},       "_2_"); // 5
    f.emplace({{1,2},   {0,1}},     "_01"); // 6
    f.emplace({{0},     {1}},       "1__"); // 7
    f.emplace({{0,1},   {0,0}},     "00_"); // 8
    f.emplace({{0,2},   {1,1}},     "1_1"); // 9
    f.emplace({{1,2},   {2,2}},     "_22"); // 10
    f.emplace({{0,1,2}, {1,1,1}},   "111"); // 11
    f.emplace({{1,2},   {2,0}},     "_20"); // 12
    f.emplace({{1,2},   {0,3}},     "_03"); // 13
    f.emplace({{0,2},   {1,2}},     "1_2"); // 14
    f.emplace({{0,2},   {1,0}},     "1_0"); // 15

    std::vector<size_t> refineIds{1, 3, 4, 7, 9, 13, 15};

    // This part will need to be upgraded to BOOST_DATA_TEST_CASE sooner or later.
    std::vector<PartialFactors> filters{
        {{2},    {0}},       // All that end with 0
        {{0, 1}, {1, 2}},    // All that begin with 1,2
        {{1},    {1}},       // All with 1 in the middle
        {{0, 1}, {1, 0}},    // All that begin with 1,0
        {{1, 2}, {0, 1}},    // All that end with 0,1
    };
    // We use this just for visual help in chekcing the test
    std::vector<std::vector<std::string>> solutions{
        {"_00", "_1_", "0__", "_2_", "1__", "00_", "_20", "1_0"},        // 2, 3, 4, 5, 7, 8, 12, 15
        {"1_3", "__2", "_2_", "1__", "1_1", "_22", "_20", "1_2", "1_0"}, // 0, 1, 5, 7, 9, 10, 12, 14, 15
        {"1_3", "__2", "_1_", "0__", "1__", "1_1", "111", "1_2", "1_0"}, // 0, 1, 3, 4, 7, 9, 11, 14, 15
        {"1_3", "__2", "_00", "_01", "1__", "1_1", "_03", "1_2", "1_0"}, // 0, 1, 2, 6, 7, 9, 13, 14, 15
        {"0__", "_01", "1__", "00_", "1_1"}                              // 4, 6, 7, 8, 9
    };

    std::vector<std::vector<size_t>> refineIdsSolutions{
        {3, 4, 7, 15},
        {1, 7, 9, 15},
        {1, 3, 4, 7, 9, 15},
        {1, 7, 9, 13, 15},
        {4, 7, 9}
    };


    for (size_t i = 0; i < filters.size(); ++i) {
        auto trie = f.getTrie();
        auto refined = trie.refine(refineIds, filters[i]);

        BOOST_TEST_INFO(i);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(refined), std::end(refined), std::begin(refineIdsSolutions[i]), std::end(refineIdsSolutions[i]));
    }
}

BOOST_AUTO_TEST_CASE( erase_id ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    FilterMap<std::string, Trie> f(F);

    f.emplace({{0,2},   {1,3}},     "1_3"); // 0
    f.emplace({{2},     {2}},       "__2"); // 1
    f.emplace({{1,2},   {0,0}},     "_00"); // 2
    f.emplace({{1},     {1}},       "_1_"); // 3
    f.emplace({{0},     {0}},       "0__"); // 4
    f.emplace({{1},     {2}},       "_2_"); // 5
    f.emplace({{1,2},   {0,1}},     "_01"); // 6
    f.emplace({{0},     {1}},       "1__"); // 7
    f.emplace({{0,1},   {0,0}},     "00_"); // 8
    f.emplace({{0,2},   {1,1}},     "1_1"); // 9
    f.emplace({{1,2},   {2,2}},     "_22"); // 10
    f.emplace({{0,1,2}, {1,1,1}},   "111"); // 11
    f.emplace({{1,2},   {2,0}},     "_20"); // 12
    f.emplace({{1,2},   {0,3}},     "_03"); // 13
    f.emplace({{0,2},   {1,2}},     "1_2"); // 14
    f.emplace({{0,2},   {1,0}},     "1_0"); // 15

    std::vector<size_t> eraseIds{1, 3, 4, 7, 9, 13, 15};

    // This part will need to be upgraded to BOOST_DATA_TEST_CASE sooner or later.
    std::vector<PartialFactors> filters{
        {{2},    {0}},       // All that end with 0
        {{0, 1}, {1, 2}},    // All that begin with 1,2
        {{1},    {1}},       // All with 1 in the middle
        {{0, 1}, {1, 0}},    // All that begin with 1,0
        {{1, 2}, {0, 1}},    // All that end with 0,1
    };
    // We use this just for visual help in chekcing the test
    std::vector<std::vector<std::string>> solutions{
        {"_00", "_1_", "0__", "_2_", "1__", "00_", "_20", "1_0"},        // 2, 3, 4, 5, 7, 8, 12, 15
        {"1_3", "__2", "_2_", "1__", "1_1", "_22", "_20", "1_2", "1_0"}, // 0, 1, 5, 7, 9, 10, 12, 14, 15
        {"1_3", "__2", "_1_", "0__", "1__", "1_1", "111", "1_2", "1_0"}, // 0, 1, 3, 4, 7, 9, 11, 14, 15
        {"1_3", "__2", "_00", "_01", "1__", "1_1", "_03", "1_2", "1_0"}, // 0, 1, 2, 6, 7, 9, 13, 14, 15
        {"0__", "_01", "1__", "00_", "1_1"}                              // 4, 6, 7, 8, 9
    };

    std::vector<std::vector<size_t>> nonErasedIdsSolutions{
        {2, 5, 8, 12},
        {0, 5, 10, 12, 14},
        {0, 11, 14},
        {0, 2, 6, 14},
        {6, 8}
    };

    for (size_t i = 0; i < filters.size(); ++i) {
        auto trie = f.getTrie();
        for (auto id : eraseIds)
            trie.erase(id);
        auto filtered = trie.filter(filters[i]);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(nonErasedIdsSolutions[i]), std::end(nonErasedIdsSolutions[i]));
    }
}

BOOST_AUTO_TEST_CASE( erase_id_pf ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    std::vector<PartialFactors> keys{
        {{0,2},   {1,3}},
        {{2},     {2}},
        {{1,2},   {0,0}},
        {{1},     {1}},
        {{0},     {0}},
        {{1},     {2}},
        {{1,2},   {0,1}},
        {{0},     {1}},
        {{0,1},   {0,0}},
        {{0,2},   {1,1}},
        {{1,2},   {2,2}},
        {{0,1,2}, {1,1,1}},
        {{1,2},   {2,0}},
        {{1,2},   {0,3}},
        {{0,2},   {1,2}},
        {{0,2},   {1,0}},
    };

    FilterMap<std::string, Trie> f(F);

    f.emplace(keys[0],   "1_3"); // 0
    f.emplace(keys[1],   "__2"); // 1
    f.emplace(keys[2],   "_00"); // 2
    f.emplace(keys[3],   "_1_"); // 3
    f.emplace(keys[4],   "0__"); // 4
    f.emplace(keys[5],   "_2_"); // 5
    f.emplace(keys[6],   "_01"); // 6
    f.emplace(keys[7],   "1__"); // 7
    f.emplace(keys[8],   "00_"); // 8
    f.emplace(keys[9],   "1_1"); // 9
    f.emplace(keys[10],   "_22"); // 10
    f.emplace(keys[11],   "111"); // 11
    f.emplace(keys[12],   "_20"); // 12
    f.emplace(keys[13],   "_03"); // 13
    f.emplace(keys[14],   "1_2"); // 14
    f.emplace(keys[15],   "1_0"); // 15

    std::vector<size_t> eraseIds{1, 3, 4, 7, 9, 13, 15};

    // This part will need to be upgraded to BOOST_DATA_TEST_CASE sooner or later.
    std::vector<PartialFactors> filters{
        {{2},    {0}},       // All that end with 0
        {{0, 1}, {1, 2}},    // All that begin with 1,2
        {{1},    {1}},       // All with 1 in the middle
        {{0, 1}, {1, 0}},    // All that begin with 1,0
        {{1, 2}, {0, 1}},    // All that end with 0,1
    };
    // We use this just for visual help in checking the test
    std::vector<std::vector<std::string>> solutions{
        {"_00", "_1_", "0__", "_2_", "1__", "00_", "_20", "1_0"},        // 2, 3, 4, 5, 7, 8, 12, 15
        {"1_3", "__2", "_2_", "1__", "1_1", "_22", "_20", "1_2", "1_0"}, // 0, 1, 5, 7, 9, 10, 12, 14, 15
        {"1_3", "__2", "_1_", "0__", "1__", "1_1", "111", "1_2", "1_0"}, // 0, 1, 3, 4, 7, 9, 11, 14, 15
        {"1_3", "__2", "_00", "_01", "1__", "1_1", "_03", "1_2", "1_0"}, // 0, 1, 2, 6, 7, 9, 13, 14, 15
        {"0__", "_01", "1__", "00_", "1_1"}                              // 4, 6, 7, 8, 9
    };

    std::vector<std::vector<size_t>> nonErasedIdsSolutions{
        {2, 5, 8, 12},
        {0, 5, 10, 12, 14},
        {0, 11, 14},
        {0, 2, 6, 14},
        {6, 8}
    };

    for (size_t i = 0; i < filters.size(); ++i) {
        auto trie = f.getTrie();
        for (auto id : eraseIds)
            trie.erase(id, keys[id]);
        auto filtered = trie.filter(filters[i]);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(nonErasedIdsSolutions[i]), std::end(nonErasedIdsSolutions[i]));
    }
}

BOOST_AUTO_TEST_CASE( construction_FT ) {
    using namespace AIToolbox::Factored;
    Factors F{1,2,3};

    FilterMap<std::string, FasterTrie> f(F);

    BOOST_CHECK(f.getF() == F);
    BOOST_CHECK_EQUAL(f.size(), 0);
}

BOOST_AUTO_TEST_CASE( filtering_FT ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    FilterMap<std::string, FasterTrie> f(F);

    f.emplace({{0,2},   {1,3}},     "1_3");
    f.emplace({{2},     {2}},       "__2");
    f.emplace({{1,2},   {0,0}},     "_00");
    f.emplace({{1},     {1}},       "_1_");
    f.emplace({{0},     {0}},       "0__");
    f.emplace({{1},     {2}},       "_2_");
    f.emplace({{1,2},   {0,1}},     "_01");
    f.emplace({{0},     {1}},       "1__");
    f.emplace({{0,1},   {0,0}},     "00_");
    f.emplace({{0,2},   {1,1}},     "1_1");
    f.emplace({{1,2},   {2,2}},     "_22");
    f.emplace({{0,1,2}, {1,1,1}},   "111");
    f.emplace({{1,2},   {2,0}},     "_20");
    f.emplace({{1,2},   {0,3}},     "_03");
    f.emplace({{0,2},   {1,2}},     "1_2");
    f.emplace({{0,2},   {1,0}},     "1_0");

    // This part will need to be upgraded to BOOST_DATA_TEST_CASE sooner or later.
    std::vector<Factors> filters{
        {0, 0, 0},
        {1, 2, 3},
        {0, 1, 2},
        {1, 0, 1},
        {0, 0, 3},
        {1, 1, 1}
    };
    std::vector<std::vector<std::string>> solutions{
        {"_00", "0__", "00_"},
        {"1_3", "_2_", "1__"},
        {"__2", "_1_", "0__"},
        {"_01", "1__", "1_1"},
        {"0__", "00_", "_03"},
        {"_1_", "1__", "1_1", "111"}
    };

    for (size_t i = 0; i < filters.size(); ++i) {
        auto filtered = f.filter(filters[i]);
        filtered.sort();
        std::sort(std::begin(solutions[i]), std::end(solutions[i]));
        BOOST_TEST_INFO(i);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(solutions[i]), std::end(solutions[i]));
    }
}

BOOST_AUTO_TEST_CASE( partial_filtering_factors_FT ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    FilterMap<std::string, FasterTrie> f(F);

    f.emplace({{0,2},   {1,3}},     "1_3");
    f.emplace({{2},     {2}},       "__2");
    f.emplace({{1,2},   {0,0}},     "_00");
    f.emplace({{1},     {1}},       "_1_");
    f.emplace({{0},     {0}},       "0__");
    f.emplace({{1},     {2}},       "_2_");
    f.emplace({{1,2},   {0,1}},     "_01");
    f.emplace({{0},     {1}},       "1__");
    f.emplace({{0,1},   {0,0}},     "00_");
    f.emplace({{0,2},   {1,1}},     "1_1");
    f.emplace({{1,2},   {2,2}},     "_22");
    f.emplace({{0,1,2}, {1,1,1}},   "111");
    f.emplace({{1,2},   {2,0}},     "_20");
    f.emplace({{1,2},   {0,3}},     "_03");
    f.emplace({{0,2},   {1,2}},     "1_2");
    f.emplace({{0,2},   {1,0}},     "1_0");

    // This part will need to be upgraded to BOOST_DATA_TEST_CASE sooner or later.
    std::vector<Factors> filters{
        {0},       // All that begin with 0
        {1, 2},    // All that begin with 1,2
        {1},       // All that begin with 1
        {1, 0},    // All that begin with 1,0
        {0, 1},    // All that begin with 0,1
    };
    std::vector<std::vector<std::string>> solutions{
        {"__2", "_00", "_1_", "0__", "_2_", "_01", "00_", "_22", "_20", "_03"},
        {"1_3", "__2", "_2_", "1__", "1_1", "_22", "_20", "1_2", "1_0"},
        {"1_3", "__2", "_00", "_1_", "_2_", "_01", "1__", "1_1", "_22", "111", "_20", "_03", "1_2", "1_0"},
        {"1_3", "__2", "_00", "_01", "1__", "1_1", "_03", "1_2", "1_0"},
        {"__2", "_1_", "0__"}
    };

    for (size_t i = 0; i < filters.size(); ++i) {
        auto filtered = f.filter(filters[i]);
        filtered.sort();
        std::sort(std::begin(solutions[i]), std::end(solutions[i]));
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(solutions[i]), std::end(solutions[i]));
    }
}

BOOST_AUTO_TEST_CASE( empty_filter_FT ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    FilterMap<std::string, FasterTrie> f(F);

    f.emplace({{0,2},   {1,3}},     "1_3");
    f.emplace({{2},     {2}},       "__2");
    f.emplace({{1,2},   {0,0}},     "_00");

    // This part will need to be upgraded to BOOST_DATA_TEST_CASE sooner or later.
    std::vector<Factors> filters{
        {0, 2, 3},
        {1, 2, 1},
        {1, 1, 0},
    };
    std::vector<std::string> solution;

    for (size_t i = 0; i < filters.size(); ++i) {
        auto filtered = f.filter(filters[i]);
        BOOST_TEST_INFO(i);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(solution), std::end(solution));
    }
}

BOOST_AUTO_TEST_CASE( erase_id_pf_FT ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    std::vector<PartialFactors> keys{
        {{0,2},   {1,3}},
        {{2},     {2}},
        {{1,2},   {0,0}},
        {{1},     {1}},
        {{0},     {0}},
        {{1},     {2}},
        {{1,2},   {0,1}},
        {{0},     {1}},
        {{0,1},   {0,0}},
        {{0,2},   {1,1}},
        {{1,2},   {2,2}},
        {{0,1,2}, {1,1,1}},
        {{1,2},   {2,0}},
        {{1,2},   {0,3}},
        {{0,2},   {1,2}},
        {{0,2},   {1,0}},
    };

    FilterMap<std::string, FasterTrie> f(F);

    f.emplace(keys[0],    "1_3"); // 0
    f.emplace(keys[1],    "__2"); // 1
    f.emplace(keys[2],    "_00"); // 2
    f.emplace(keys[3],    "_1_"); // 3
    f.emplace(keys[4],    "0__"); // 4
    f.emplace(keys[5],    "_2_"); // 5
    f.emplace(keys[6],    "_01"); // 6
    f.emplace(keys[7],    "1__"); // 7
    f.emplace(keys[8],    "00_"); // 8
    f.emplace(keys[9],    "1_1"); // 9
    f.emplace(keys[10],   "_22"); // 10
    f.emplace(keys[11],   "111"); // 11
    f.emplace(keys[12],   "_20"); // 12
    f.emplace(keys[13],   "_03"); // 13
    f.emplace(keys[14],   "1_2"); // 14
    f.emplace(keys[15],   "1_0"); // 15

    std::vector<Factors> filters{
        {0},       // All that begin with 0
        {1, 2},    // All that begin with 1,2
        {1},       // All that begin with 1
        {1, 0},    // All that begin with 1,0
        {0, 1},    // All that begin with 0,1
    };
    std::vector<std::vector<std::string>> solutions{
        // 1      2      3      4      5      6      8      10     12     13
        {"__2", "_00", "_1_", "0__", "_2_", "_01", "00_", "_22", "_20", "_03"},
        // 0      1      5      7      9      10     12     14     15
        {"1_3", "__2", "_2_", "1__", "1_1", "_22", "_20", "1_2", "1_0"},
        // 0      1      2      3      5      6      7      9      10     11     12     13     14     15
        {"1_3", "__2", "_00", "_1_", "_2_", "_01", "1__", "1_1", "_22", "111", "_20", "_03", "1_2", "1_0"},
        // 0      1      2      6      7      9      13     14     15
        {"1_3", "__2", "_00", "_01", "1__", "1_1", "_03", "1_2", "1_0"},
        // 1      3      4
        {"__2", "_1_", "0__"}
    };

    std::vector<size_t> eraseIds{1, 3, 4, 7, 9, 13, 15};

    std::vector<std::vector<size_t>> nonErasedIdsSolutions{
        {2, 5, 6, 8, 10, 12},
        {0, 5, 10, 12, 14},
        {0, 2, 5, 6, 10, 11, 12, 14},
        {0, 2, 6, 14},
        {}
    };

    for (size_t i = 0; i < filters.size(); ++i) {
        auto trie = f.getTrie();
        for (auto id : eraseIds)
            trie.erase(id, keys[id]);
        auto filtered = trie.filter(filters[i]);
        std::sort(std::begin(filtered), std::end(filtered));
        std::sort(std::begin(nonErasedIdsSolutions[i]), std::end(nonErasedIdsSolutions[i]));
        BOOST_TEST_INFO(i);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(nonErasedIdsSolutions[i]), std::end(nonErasedIdsSolutions[i]));
    }
}

BOOST_AUTO_TEST_CASE( reconstruction_FT ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    std::vector<PartialFactors> keys{
        {{0,2},   {1,3}},
        {{2},     {2}},
        {{1,2},   {0,0}},
        {{1},     {1}},
        {{0},     {0}},
        {{1},     {2}},
        {{1,2},   {0,1}},
        {{0},     {1}},
        {{0,1},   {0,0}},
        {{0,2},   {1,1}},
        {{1,2},   {2,2}},
        {{0,1,2}, {1,1,1}},
        {{1,2},   {2,0}},
        {{1,2},   {0,3}},
        {{0,2},   {1,2}},
        {{0,2},   {1,0}},
    };

    std::vector<std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>> solutions {
        {
            {{1, 4, 8},     {0, 0, 2}},
            {{1, 3, 4},     {0, 1, 2}},
            {{1, 4, 5, 10}, {0, 2, 2}}
        },
        {
            {{3, 7, 15},    {1, 1, 0}},
            {{3, 7, 9, 11}, {1, 1, 1}},
            {{1, 3, 7, 14}, {1, 1, 2}},
            {{0, 3, 7},     {1, 1, 3}}
        },
        {
            {{2, 4, 8},     {0, 0, 0}},
            {{2, 7, 15},    {1, 0, 0}},
            {{3, 4},        {0, 1, 0}},
            {{3, 7, 15},    {1, 1, 0}},
            {{4, 5, 12},    {0, 2, 0}},
            {{5, 7, 12, 15}, {1, 2, 0}}
        }
    };

    std::vector<PartialFactors> startKeys {
        {{0,2},{0,2}}, // "0_2"
        {{0,1},{1,1}}, // "11_"
        {{2}, {0}},    // "__0"
    };

    for (size_t i = 0; i < startKeys.size(); ++i) {
        FasterTrie trie(F);
        for (const auto key : keys)
            trie.insert(key);

        std::vector<unsigned> counts(solutions[i].size());
        for (size_t j = 0; j < 1000; ++j) {
            auto [ids, factor] = trie.reconstruct(startKeys[i]);
            std::sort(std::begin(ids), std::end(ids));

            for (size_t k = 0; k < counts.size(); ++k) {
                if (AIToolbox::veccmp(factor, solutions[i][k].second) == 0) {
                    for (size_t q = 0; q < ids.size(); ++q) {
                        BOOST_CHECK_EQUAL(ids[q].first, solutions[i][k].first[q]);
                        BOOST_CHECK(AIToolbox::veccmp(ids[q].second.first,  keys[ids[q].first].first) == 0);
                        BOOST_CHECK(AIToolbox::veccmp(ids[q].second.second, keys[ids[q].first].second) == 0);
                    }
                    ++counts[k];
                    break;
                }
            }
        }

        // Not sure how to compute the % each solution should be found, so here
        // we just check that at least all of them have been found once.
        for (size_t k = 0; k < counts.size(); ++k)
            BOOST_CHECK(counts[k] > 0);
    }
}

BOOST_AUTO_TEST_CASE( reconstruction_removal_FT ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    std::vector<PartialFactors> keys{
        {{0,2},   {1,3}},   // "1_3"  0
        {{2},     {2}},     // "__2"  1
        {{1,2},   {0,0}},   // "_00"  2
        {{1},     {1}},     // "_1_"  3
        {{0},     {0}},     // "0__"  4
        {{1},     {2}},     // "_2_"  5
        {{1,2},   {0,1}},   // "_01"  6
        {{0},     {1}},     // "1__"  7
        {{0,1},   {0,0}},   // "00_"  8
        {{0,2},   {1,1}},   // "1_1"  9
        {{1,2},   {2,2}},   // "_22"  10
        {{0,1,2}, {1,1,1}}, // "111"  11
        {{1,2},   {2,0}},   // "_20"  12
        {{1,2},   {0,3}},   // "_03"  13
        {{0,2},   {1,2}},   // "1_2"  14
        {{0,2},   {1,0}},   // "1_0"  15
    };

    std::vector<PartialFactors> reconstructions {
        {{0,1,2},{1,1,1}},  // "111" -> "111" _1_ (3), 1__ (7), 1_1 (9), 111 (11)
        {{0,2},{1,2}},      // "1_2" -> "122" __2 (1), _2_ (5), _22 (10), 1_2 (14)
        {{0,2},{0,3}},      // "0_3" -> "003" 0__ (4), 00_ (8), _03 (13)
        {{2},{1}},          // "__1" -> "_01" _01 (6)
        {{1,2},{1,3}},      // "_13" -> "113" 1_3 (0)
        {{1},{2}},          // "_2_" -> "120" _20 (12), 1_0 (15)
        {{2},{0}},          // "__0" -> "_00" _00 (2)
    };

    std::vector<std::tuple<std::vector<size_t>, Factors>> solutions {
        {{3, 7, 9, 11}, {1,1,1}},
        {{1, 5, 10, 14}, {1,2,2}},
        {{4, 8, 13}, {0,0,3}},
        {{6}, {2,0,1}},
        {{0}, {1,1,3}},
        {{12, 15}, {1,2,0}},
        {{2}, {2,0,0}},
    };

    FasterTrie trie(F);
    for (const auto key : keys)
        trie.insert(key);

    for (size_t i = 0; i < reconstructions.size(); ++i) {
        auto [ids, factor] = trie.reconstruct(reconstructions[i], true);
        auto [sids, sfactor] = solutions[i];

        std::sort(std::begin(ids), std::end(ids));

        for (size_t q = 0; q < ids.size(); ++q) {
            BOOST_CHECK_EQUAL(ids[q].first,   sids[q]);
            BOOST_CHECK(AIToolbox::veccmp(ids[q].second.first,  keys[ids[q].first].first) == 0);
            BOOST_CHECK(AIToolbox::veccmp(ids[q].second.second, keys[ids[q].first].second) == 0);
        }

        BOOST_CHECK(AIToolbox::veccmp(factor, sfactor) == 0);
    }
}
