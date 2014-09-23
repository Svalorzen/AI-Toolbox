#define BOOST_TEST_MODULE MDP_Model
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/IO.hpp>
#include <AIToolbox/MDP/Model.hpp>

#include <fstream>

BOOST_AUTO_TEST_CASE( construction ) {
    const int S = 5, A = 6;

    AIToolbox::MDP::Model m(S, A);

    BOOST_CHECK_EQUAL(m.getS(), S);
    BOOST_CHECK_EQUAL(m.getA(), A);

    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,0,0), 1.0);
    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,1,0), 1.0);
    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,0,1), 0.0);
    BOOST_CHECK_EQUAL(m.getTransitionProbability(0,1,1), 0.0);

    BOOST_CHECK_EQUAL(m.getExpectedReward(0,0,0), 0.0);
}

int generator() {
    static int counter = 0;
    return ++counter;
}

BOOST_AUTO_TEST_CASE( files ) {
    const int S = 4, A = 2;
    AIToolbox::MDP::Model m(S,A);

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
        std::ifstream inputFile(inputFilename);
        std::ifstream writtenFile(outputFilename);

        double input, written;
        while ( inputFile >> input ) {
            BOOST_CHECK( writtenFile >> written );
            BOOST_CHECK_EQUAL( written, input );
        }
        BOOST_CHECK( ! ( writtenFile >> written ) );
    }
    // Cleanup
    {
        std::remove(outputFilename.c_str());
    }
}
