#define BOOST_TEST_MODULE MDP_Model
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/IO.hpp>
#include <AIToolbox/MDP/Model.hpp>

#include <AIToolbox/MDP/Environments/CornerProblem.hpp>

#include <fstream>

BOOST_AUTO_TEST_CASE( eigen_model ) {
    BOOST_CHECK(AIToolbox::MDP::is_model_eigen_v<AIToolbox::MDP::Model>);
}

BOOST_AUTO_TEST_CASE( construction ) {
    const size_t S = 5, A = 6;

    AIToolbox::MDP::Model m(S, A);

    BOOST_CHECK_EQUAL(m.getS(), S);
    BOOST_CHECK_EQUAL(m.getA(), A);

    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,0,0), 1.0);
    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,1,0), 1.0);
    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,0,1), 0.0);
    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,1,1), 0.0);

    BOOST_CHECK_EQUAL(m.getExpectedReward(0,0,0), 0.0);

    for (size_t s = 0; s < S; ++s)
        BOOST_CHECK(m.isTerminal(s));
}

BOOST_AUTO_TEST_CASE( copy_construction ) {
    using namespace AIToolbox::MDP;

    GridWorld grid(4, 4);

    auto model = makeCornerProblem(grid);
    size_t S = model.getS(), A = model.getA();

    Model copy(model);

    BOOST_CHECK_EQUAL(model.getDiscount(), copy.getDiscount());
    BOOST_CHECK_EQUAL(S, copy.getS());
    BOOST_CHECK_EQUAL(A, copy.getA());

    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                BOOST_CHECK_EQUAL(model.getTransitionProbability(s, a, s1), copy.getTransitionProbability(s, a, s1));
                BOOST_CHECK_EQUAL(model.getExpectedReward(s, a, s1), copy.getExpectedReward(s, a, s1));
            }
        }
    }
}

int generator() {
    static int counter = 0;
    return ++counter;
}

BOOST_AUTO_TEST_CASE( files ) {
    const size_t S = 4, A = 2;
    AIToolbox::MDP::Model m(S,A), m2(S,A);

    std::string inputFilename  = "./data/mdp_model.txt";
    std::string outputFilename = "./loadedModel.txt";
    {
        std::ifstream inputFile(inputFilename);

        if ( !inputFile ) BOOST_FAIL("Data to perform test could not be loaded: " + inputFilename);
        BOOST_CHECK( inputFile >> m );
    }
    {
        std::ofstream outputFile(outputFilename);
        if ( !outputFile ) BOOST_FAIL("Could not open file for writing: " + outputFilename);
        BOOST_CHECK( outputFile << m );
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
            }
        }
    }
    // Cleanup
    {
        std::remove(outputFilename.c_str());
    }
}

BOOST_AUTO_TEST_CASE( cassandraCorner ) {
    using namespace AIToolbox::MDP;

    GridWorld grid(2, 2);

    auto m = makeCornerProblem(grid);
    size_t S = m.getS(), A = m.getA();

    std::string inputFilename  = "./data/corner.MDP";

    std::ifstream inputFile(inputFilename);
    if ( !inputFile ) BOOST_FAIL("Data to perform test could not be loaded: " + inputFilename);

    auto m2 = AIToolbox::MDP::parseCassandra(inputFile);

    BOOST_CHECK_EQUAL(m.getS(), m2.getS());
    BOOST_CHECK_EQUAL(m.getA(), m2.getA());

    for ( size_t a = 0; a < A; ++a )
    for ( size_t s = 0; s < S; ++s )
    for ( size_t s1 = 0; s1 < S; ++s1 ) {
        BOOST_CHECK(AIToolbox::checkEqualSmall(m.getTransitionProbability(s, a, s1), m2.getTransitionProbability(s, a, s1)));
        BOOST_CHECK(AIToolbox::checkEqualGeneral(m.getExpectedReward(s, a, s1), m2.getExpectedReward(s, a, s1)));
    }
}
