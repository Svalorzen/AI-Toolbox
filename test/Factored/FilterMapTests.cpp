#define BOOST_TEST_MODULE Factored_FactoredContainer
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Factored/Utils/FactoredContainer.hpp>
#include <string>

BOOST_AUTO_TEST_CASE( construction ) {
    using namespace AIToolbox::Factored;
    Factors F{1,2,3};

    FactoredContainer<std::string> f(F);

    BOOST_CHECK(f.getF() == F);
    BOOST_CHECK_EQUAL(f.size(), 0);
}

BOOST_AUTO_TEST_CASE( filtering ) {
    using namespace AIToolbox::Factored;
    Factors F{2,3,4};

    FactoredContainer<std::string> f(F);

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

    FactoredContainer<std::string> f(F);

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

    FactoredContainer<std::string> f(F);

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

    FactoredContainer<std::string> f(F);

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

    FactoredContainer<std::string> f(F);

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

    FactoredContainer<std::string> f(F);

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

    FactoredContainer<std::string> f(F);

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
            trie.erase(id, keys[id]);
        auto filtered = trie.filter(filters[i]);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(filtered), std::end(filtered), std::begin(nonErasedIdsSolutions[i]), std::end(nonErasedIdsSolutions[i]));
    }
}
