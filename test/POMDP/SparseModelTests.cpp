#define BOOST_TEST_MODULE POMDP_Model
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/IO.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

#include <AIToolbox/POMDP/Environments/TigerProblem.hpp>

#include <fstream>

BOOST_AUTO_TEST_CASE( construction ) {
    using namespace AIToolbox;
    const size_t S = 5, A = 6, O = 2;

    POMDP::SparseModel<MDP::Model> m(O, S, A);

    BOOST_CHECK_EQUAL(m.getS(), S);
    BOOST_CHECK_EQUAL(m.getA(), A);
    BOOST_CHECK_EQUAL(m.getO(), O);

    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,0,0), 1.0);
    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,1,0), 1.0);
    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,0,1), 0.0);
    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,1,1), 0.0);

    BOOST_CHECK_EQUAL(m.getExpectedReward(0,0,0), 0.0);

    BOOST_CHECK_EQUAL(m.getObservationProbability(0,0,0), 1.0);
    BOOST_CHECK_EQUAL(m.getObservationProbability(0,1,0), 1.0);
    BOOST_CHECK_EQUAL(m.getObservationProbability(0,0,1), 0.0);
    BOOST_CHECK_EQUAL(m.getObservationProbability(0,1,1), 0.0);
}

BOOST_AUTO_TEST_CASE( other_construction ) {
    using namespace AIToolbox;
    const size_t S = 5, A = 6, O = 2;

    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a )
            transitions[s][a][s] = 1.0;

    for ( size_t s1 = 0; s1 < S; ++s1 )
        for ( size_t a = 0; a < A; ++a )
            observations[s1][a][0] = 1.0;

    POMDP::SparseModel<MDP::Model> m(O, observations, S, A, transitions, rewards);
}

BOOST_AUTO_TEST_CASE( copy_construction ) {
    using namespace AIToolbox;

    auto model = POMDP::makeTigerProblem();

    POMDP::Model<MDP::Model> copy(model);

    size_t S = model.getS(), A = model.getA(), O = model.getO();

    BOOST_CHECK_EQUAL(model.getDiscount(), copy.getDiscount());
    BOOST_CHECK_EQUAL(S, copy.getS());
    BOOST_CHECK_EQUAL(A, copy.getA());
    BOOST_CHECK_EQUAL(O, copy.getO());

    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                BOOST_CHECK_EQUAL(model.getTransitionProbability(s, a, s1), copy.getTransitionProbability(s, a, s1));
                BOOST_CHECK_EQUAL(model.getExpectedReward(s, a, s1), copy.getExpectedReward(s, a, s1));
            }
            for ( size_t o = 0; o < O; ++o ) {
                BOOST_CHECK_EQUAL(model.getObservationProbability(s, a, o), copy.getObservationProbability(s, a, o));
            }
        }
    }
}

int generator() {
    static int counter = 0;
    return ++counter;
}

BOOST_AUTO_TEST_CASE( files ) {
    using namespace AIToolbox;
    const size_t S = 4, A = 2, O = 2;

    POMDP::SparseModel<MDP::Model> m(O, S, A), m2(O, S, A);

    std::string inputFilename  = "./data/pomdp_model.txt";
    std::string outputFilename = "./loadedModel.txt";
    {
        std::ifstream inputFile(inputFilename);

        if ( !inputFile ) BOOST_FAIL("Data to perform test could not be loaded: " + inputFilename);
        BOOST_CHECK( POMDP::operator>>(inputFile, m) );
    }
    {
        std::ofstream outputFile(outputFilename);
        if ( !outputFile ) BOOST_FAIL("Could not open file for writing: " + outputFilename);
        BOOST_CHECK( POMDP::operator<<(outputFile, m) );
    }
    {
        std::ifstream inputFile(outputFilename);

        if ( !inputFile ) BOOST_FAIL("Data written cannot be read again: " + inputFilename);
        BOOST_CHECK( inputFile >> m2 );
    }
    {
    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                BOOST_CHECK(AIToolbox::checkEqualSmall(m.getTransitionProbability(s, a, s1), m2.getTransitionProbability(s, a, s1)));
                BOOST_CHECK(AIToolbox::checkEqualGeneral(m.getExpectedReward(s, a, s1), m2.getExpectedReward(s, a, s1)));
            }
            for ( size_t o = 0; o < O; ++o ) {
                BOOST_CHECK(AIToolbox::checkEqualSmall(m.getObservationProbability(s, a, o), m2.getObservationProbability(s, a, o)));
            }
        }
    }
    }
    // Cleanup
    {
        std::remove(outputFilename.c_str());
    }
}
