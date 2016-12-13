#define BOOST_TEST_MODULE FactoredMDP_FactoredContainer
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/FactoredMDP/FactoredContainer.hpp>
#include <string>

BOOST_AUTO_TEST_CASE( construction ) {
    using namespace AIToolbox::FactoredMDP;
    State S{1,2,3};

    FactoredContainer<std::string> f(S);

    BOOST_CHECK(f.getS() == S);
    BOOST_CHECK_EQUAL(f.size(), 0);
}

BOOST_AUTO_TEST_CASE( filtering ) {
    using namespace AIToolbox::FactoredMDP;
    State S{2,3,4};

    FactoredContainer<std::string> f(S);

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
    std::vector<State> filters{
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
