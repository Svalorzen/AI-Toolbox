#ifndef AI_TOOLBOX_TEST_GLOBAL_FIXTURES_HEADER_FILE

#include <boost/test/unit_test.hpp>
#include <boost/test/results_collector.hpp>
#include <boost/test/tree/visitor.hpp>
#include <boost/test/tree/traverse.hpp>

#include <AIToolbox/Impl/Seeder.hpp>

struct SeedPrinter {
    struct AllPassVisitor : boost::unit_test::test_tree_visitor {
        bool passed = true;

        void visit( boost::unit_test::test_case const& test ) {
            passed &= boost::unit_test::results_collector.results(test.p_id).passed();
        }
    };

    ~SeedPrinter() {
        namespace ut = boost::unit_test;

        // Check whether we have passed all tests.
        AllPassVisitor v;
        ut::traverse_test_tree(ut::framework::master_test_suite(), v);

        // If not, print the global seed to reproduce errors.
        if (!v.passed)
            BOOST_CHECK_MESSAGE(false, "ROOT SEED: " << AIToolbox::Impl::Seeder::getRootSeed());
    }
};

BOOST_TEST_GLOBAL_FIXTURE(SeedPrinter);

#endif
