#define BOOST_TEST_MODULE MDP_RLModel
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/IO.hpp>

#include <fstream>

BOOST_AUTO_TEST_CASE( construction ) {
    const int S = 10, A = 8;

    AIToolbox::Experience exp(S,A);
    AIToolbox::MDP::RLModel model(exp, false);

    for ( size_t s = 0; s < S; ++s )
        for ( size_t s1 = 0; s1 < S; ++s1 )
            for ( size_t a = 0; a < A; ++a ) {
                if ( s != s1 )  BOOST_CHECK_EQUAL( model.getTransitionProbability(s,s1,a), 0.0 );
                else            BOOST_CHECK_EQUAL( model.getTransitionProbability(s,s1,a), 1.0 );
                BOOST_CHECK_EQUAL( model.getExpectedReward(s,s1,a), 0.0 );
            }
}

BOOST_AUTO_TEST_CASE( syncing ) {
    const int S = 10, A = 8;

    AIToolbox::Experience exp(S,A);
    // Single state sync
    {
        AIToolbox::MDP::RLModel model(exp, false);

        exp.record(0,1,0,10);
        exp.record(0,2,0,10);
        exp.record(0,3,0,10);

        exp.record(4,5,0,10);

        BOOST_CHECK_EQUAL( model.getTransitionProbability(0,1,0), 0.0 ); // Not yet synced
        BOOST_CHECK_EQUAL( model.getTransitionProbability(0,0,0), 1.0 ); // Not yet synced
        BOOST_CHECK_EQUAL( model.getExpectedReward(0,1,0), 0.0 );

        model.sync(0,0);
        BOOST_CHECK_EQUAL( model.getTransitionProbability(0,1,0), 1.0/3.0 );
        BOOST_CHECK_EQUAL( model.getTransitionProbability(0,2,0), 1.0/3.0 );
        BOOST_CHECK_EQUAL( model.getTransitionProbability(0,4,0), 0.0 );

        BOOST_CHECK_EQUAL( model.getExpectedReward(0,1,0), 10.0 );
        BOOST_CHECK_EQUAL( model.getExpectedReward(0,1,0), 10.0 );
        BOOST_CHECK_EQUAL( model.getExpectedReward(0,1,0), 10.0 );
        BOOST_CHECK_EQUAL( model.getExpectedReward(0,4,0), 0.0 );

        BOOST_CHECK_EQUAL( model.getTransitionProbability(4,5,0), 0.0 ); // Not yet synced
        BOOST_CHECK_EQUAL( model.getExpectedReward(4,5,0),        0.0 ); // Not yet synced

        model.sync(4,0);
        BOOST_CHECK_EQUAL( model.getTransitionProbability(4,5,0), 1.0  ); // Now it is
        BOOST_CHECK_EQUAL( model.getExpectedReward(4,5,0),        10.0 ); // Not yet synced

    }
    // Full sync, manual or on construction
    {
        AIToolbox::MDP::RLModel model(exp, false);
        model.sync();

        AIToolbox::MDP::RLModel model2(exp, true);

        BOOST_CHECK_EQUAL( model.getTransitionProbability (0,1,0), 1.0/3.0 );
        BOOST_CHECK_EQUAL( model2.getTransitionProbability(0,1,0), 1.0/3.0 );

        BOOST_CHECK_EQUAL( model.getTransitionProbability(4,5,0), 1.0 );
        BOOST_CHECK_EQUAL( model2.getTransitionProbability(4,5,0), 1.0 );

        // Check multiple rewards are averaged correctly
        exp.record(0,1,0,50);
        model.sync(0,0);

        BOOST_CHECK_EQUAL( model.getExpectedReward(0,1,0), 30.0 );
    }
}

/*
BOOST_AUTO_TEST_CASE( IO ) {
    const int S = 10, A = 8;

    std::string inputExpFilename     = "./data/experience.txt";
    std::string inputModelFilename   = "./data/model.txt";
    std::string outputFilename       = "./computedModule.txt";

    AIToolbox::Experience exp(S,A);
    {
        std::ifstream inputExpFile(inputExpFilename);

        if ( !inputExpFile ) BOOST_FAIL("Data to perform test could not be loaded: " + inputExpFilename);
        BOOST_CHECK( inputExpFile >> exp );
    }

    AIToolbox::MDP::RLModel model(exp, true);
    {
        std::ofstream outputFile(outputFilename);

        if ( !outputFile ) BOOST_FAIL("Could not open file for writing: " + outputFilename);
        BOOST_CHECK( outputFile << model );
    }
    {
        std::ifstream inputModelFile(inputModelFilename);
        if ( !inputModelFile ) BOOST_FAIL("Data to perform test could not be loaded: " + inputModelFilename);
        std::ifstream writtenData(outputFilename);

        std::stringstream truthBuffer, writtenBuffer;

        truthBuffer     << inputModelFile.rdbuf();
        writtenBuffer   << writtenData.rdbuf();

        BOOST_CHECK( truthBuffer.str() == writtenBuffer.str() );
    }
    // Cleanup
    {
        std::remove(outputFilename.c_str());
    }
}
*/
