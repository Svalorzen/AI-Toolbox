#define BOOST_TEST_MODULE UtilsCore
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/IndexMap.hpp>
#include <AIToolbox/Utils/Combinatorics.hpp>

#include <boost/iterator/indirect_iterator.hpp>

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

BOOST_AUTO_TEST_CASE( index_map ) {
    std::vector<std::string> test{"aaa", "bbb", "ccc", "ddd"};
    std::vector<std::vector<size_t>> lists {
        {},
        {3},
        {0, 1},
        {0, 3},
        {1, 2},
        {0, 2, 3},
        {0, 1, 2, 3}
    };
    std::vector<std::vector<std::string>> solutions {
        {},
        {"ddd"},
        {"aaa", "bbb"},
        {"aaa", "ddd"},
        {"bbb", "ccc"},
        {"aaa", "ccc", "ddd"},
        {"aaa", "bbb", "ccc", "ddd"},
    };

    for (size_t i = 0; i < lists.size(); ++i) {
        AIToolbox::IndexMap e(lists[i], test);

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(e), std::end(e),
                                      std::begin(solutions[i]), std::end(solutions[i]));
        // Test random iterators.
        constexpr auto v1 = std::is_same_v<std::iterator_traits<decltype(e)::iterator>::iterator_category, std::random_access_iterator_tag>;
        constexpr auto v2 = std::is_same_v<std::iterator_traits<decltype(e)::const_iterator>::iterator_category, std::random_access_iterator_tag>;
        BOOST_CHECK(v1);
        BOOST_CHECK(v2);

        (void)(std::begin(e) + 3);
    }
}

BOOST_AUTO_TEST_CASE( index_skip_map ) {
    std::vector<std::string> test{"aaa", "bbb", "ccc", "ddd"};
    std::vector<std::vector<size_t>> lists {
        {},
        {3},
        {0, 1},
        {0, 3},
        {1, 2},
        {0, 2, 3},
        {0, 1, 2, 3}
    };
    std::vector<std::vector<std::string>> solutions {
        {"aaa", "bbb", "ccc", "ddd"},
        {"aaa", "bbb", "ccc"},
        {"ccc", "ddd"},
        {"bbb", "ccc"},
        {"aaa", "ddd"},
        {"bbb"},
        {}
    };

    for (size_t i = 0; i < lists.size(); ++i) {
        AIToolbox::IndexSkipMap e(lists[i], test);

        BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(e), std::end(e),
                                      std::begin(solutions[i]), std::end(solutions[i]));
    }
}

BOOST_AUTO_TEST_CASE( subset_enumeration_number ) {
    std::vector<std::vector<int>> solutions {
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3}
    };
    constexpr size_t size = 2;

    AIToolbox::SubsetEnumerator e(size, 0, 4);
    auto begin = std::begin(*e),
         end   = std::end(*e);

    unsigned counter = 0;
    for (; e.isValid(); e.advance(), ++counter) {
        BOOST_CHECK_EQUAL_COLLECTIONS(begin, end,
                                      std::begin(solutions[counter]), std::end(solutions[counter]));
    }
    BOOST_CHECK_EQUAL(counter, e.subsetsSize());
    BOOST_CHECK_EQUAL(e.subsetsSize(), AIToolbox::nChooseK(4, size));
    BOOST_CHECK_EQUAL(solutions.size(), counter);
}

BOOST_AUTO_TEST_CASE( subset_enumeration_it ) {
    std::vector<std::string> test{"aaa", "bbb", "ccc", "ddd"};
    std::vector<std::vector<std::string>> solutions {
        {"aaa", "bbb"},
        {"aaa", "ccc"},
        {"aaa", "ddd"},
        {"bbb", "ccc"},
        {"bbb", "ddd"},
        {"ccc", "ddd"},
    };
    constexpr size_t size = 2;

    AIToolbox::SubsetEnumerator e(size, std::begin(test), std::end(test));
    auto begin = boost::make_indirect_iterator(std::begin(*e)),
         end   = boost::make_indirect_iterator(std::end(*e));

    unsigned counter = 0;
    for (; e.isValid(); e.advance(), ++counter) {
        BOOST_CHECK_EQUAL_COLLECTIONS(begin, end,
                                      std::begin(solutions[counter]), std::end(solutions[counter]));
    }
    BOOST_CHECK_EQUAL(counter, e.subsetsSize());
    BOOST_CHECK_EQUAL(e.subsetsSize(), AIToolbox::nChooseK(test.size(), size));
    BOOST_CHECK_EQUAL(solutions.size(), counter);
}
