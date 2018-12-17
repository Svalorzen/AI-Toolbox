#define BOOST_TEST_MODULE MDP_SparseRLModel
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>

// #include <AIToolbox/MDP/IO.hpp>
// #include <fstream>

BOOST_AUTO_TEST_CASE( eigen_model ) {
    BOOST_CHECK(AIToolbox::MDP::is_model_eigen_v<AIToolbox::MDP::SparseRLModel<AIToolbox::MDP::SparseExperience>>);
}

BOOST_AUTO_TEST_CASE( construction ) {
    const size_t S = 10, A = 8;

    AIToolbox::MDP::SparseExperience exp(S,A);
    AIToolbox::MDP::SparseRLModel model(exp, 1.0, false);

    for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a )
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                if ( s != s1 )  BOOST_CHECK_EQUAL( model.getTransitionProbability(s,a,s1), 0.0 );
                else            BOOST_CHECK_EQUAL( model.getTransitionProbability(s,a,s1), 1.0 );
                BOOST_CHECK_EQUAL( model.getExpectedReward(s,a,s1), 0.0 );
            }
}

BOOST_AUTO_TEST_CASE( syncing ) {
    const size_t S = 10, A = 8;

    AIToolbox::MDP::SparseExperience exp(S,A);
    // Single state sync
    {
        AIToolbox::MDP::SparseRLModel model(exp, 1.0, false);

        exp.record(0,0,1,10);
        exp.record(0,0,2,10);
        exp.record(0,0,3,10);

        exp.record(4,0,5,10);

        BOOST_CHECK_EQUAL( model.getTransitionProbability(0,0,1), 0.0 ); // Not yet synced
        BOOST_CHECK_EQUAL( model.getTransitionProbability(0,0,0), 1.0 ); // Not yet synced
        BOOST_CHECK_EQUAL( model.getExpectedReward(0,0,1), 0.0 );

        model.sync(0,0);
        BOOST_CHECK_EQUAL( model.getTransitionProbability(0,0,1), 1.0/3.0 );
        BOOST_CHECK_EQUAL( model.getTransitionProbability(0,0,2), 1.0/3.0 );
        BOOST_CHECK_EQUAL( model.getTransitionProbability(0,0,4), 0.0 );

        BOOST_CHECK_EQUAL( model.getExpectedReward(0,0,1), 10.0 );
        BOOST_CHECK_EQUAL( model.getExpectedReward(0,0,1), 10.0 );
        BOOST_CHECK_EQUAL( model.getExpectedReward(0,0,1), 10.0 );
        BOOST_CHECK_EQUAL( model.getExpectedReward(0,0,4), 10.0 ); // Wasn't recorded, but by storing S,A we get this.

        BOOST_CHECK_EQUAL( model.getTransitionProbability(4,0,5), 0.0 ); // Not yet synced
        BOOST_CHECK_EQUAL( model.getExpectedReward(4,0,5),        0.0 ); // Not yet synced

        model.sync(4,0);
        BOOST_CHECK_EQUAL( model.getTransitionProbability(4,0,5), 1.0  ); // Now it is
        BOOST_CHECK_EQUAL( model.getExpectedReward(4,0,5),        10.0 ); // Not yet synced

    }
    // Full sync, manual or on construction
    {
        AIToolbox::MDP::SparseRLModel<decltype(exp)>  model(exp, 1.0, false);
        model.sync();

        AIToolbox::MDP::SparseRLModel<decltype(exp)>  model2(exp, 1.0, true);

        BOOST_CHECK_EQUAL( model.getTransitionProbability (0,0,1), 1.0/3.0 );
        BOOST_CHECK_EQUAL( model2.getTransitionProbability(0,0,1), 1.0/3.0 );

        BOOST_CHECK_EQUAL( model.getTransitionProbability (4,0,5), 1.0 );
        BOOST_CHECK_EQUAL( model2.getTransitionProbability(4,0,5), 1.0 );

        // Check multiple rewards are averaged correctly
        exp.record(0,0,1,50);
        model.sync(0,0);

        BOOST_CHECK_EQUAL( model.getExpectedReward(0,0,1), (30.0 + 50.0) / 4 );
    }
}

BOOST_AUTO_TEST_CASE( syncing_rew_to_zero ) {
    const size_t S = 10, A = 8;

    AIToolbox::MDP::SparseExperience exp(S,A);

    AIToolbox::MDP::SparseRLModel model(exp, 1.0, false);

    exp.record(0,0,1,10);
    model.sync();

    exp.record(0,1,2,10);
    model.sync(0, 1);

    exp.record(0,2,3,10);
    model.sync(0, 2, 3);

    BOOST_CHECK_EQUAL( model.getExpectedReward(0,0,1), 10.0 );
    BOOST_CHECK_EQUAL( model.getExpectedReward(0,1,2), 10.0 );
    BOOST_CHECK_EQUAL( model.getExpectedReward(0,2,3), 10.0 );

    exp.record(0,0,1,-10);
    model.sync();

    exp.record(0,1,2,-10);
    model.sync(0, 1);

    exp.record(0,2,3,-10);
    model.sync(0, 2, 3);

    BOOST_CHECK_EQUAL( model.getExpectedReward(0,0,1), 0.0 );
    BOOST_CHECK_EQUAL( model.getExpectedReward(0,1,2), 0.0 );
    BOOST_CHECK_EQUAL( model.getExpectedReward(0,2,3), 0.0 );
}

BOOST_AUTO_TEST_CASE( clearInitialTransition ) {
    const size_t S = 2, A = 2;

    AIToolbox::MDP::SparseExperience exp(S,A);
    AIToolbox::MDP::SparseRLModel model(exp, 1.0, false);

    exp.record(0,0,1,10);
    model.sync(0,0);

    BOOST_CHECK_EQUAL( model.getTransitionProbability(0,0,0), 0.0 );
    BOOST_CHECK_EQUAL( model.getTransitionProbability(0,0,1), 1.0 );

    BOOST_CHECK_EQUAL( model.getTransitionProbability(0,1,0), 1.0 );
    BOOST_CHECK_EQUAL( model.getTransitionProbability(0,1,1), 0.0 );
}

BOOST_AUTO_TEST_CASE( sampling ) {
    const size_t S = 10, A = 8;

    AIToolbox::MDP::SparseExperience exp(S,A);
    AIToolbox::MDP::SparseRLModel model(exp, 1.0, false);

    exp.record(0,0,0,0);
    exp.record(0,0,1,0);

    exp.record(1,1,2,0);
    exp.record(2,2,5,0);
    exp.record(5,1,0,5.0);

    model.sync();

    for ( int i = 0; i < 1000; ++i )
        BOOST_CHECK_EQUAL( std::get<1>(model.sampleSR(5,1)), 5.0 );

    unsigned k = 0;
    for ( int i = 0; i < 10000; ++i )
        if ( std::get<0>(model.sampleSR(0,0)) == 1 ) ++k;

    BOOST_CHECK_MESSAGE( k > 4000 && k < 6000, "This test may fail from time to time as it is based on sampling. k should be ~5000. k is " << k ); // Hopefully

    exp.record(0,0,0,0);
    model.sync(0,0);

    k = 0;
    for ( int i = 0; i < 10000; ++i )
        if ( std::get<0>(model.sampleSR(0,0)) == 1 ) ++k;

    BOOST_CHECK_MESSAGE( k > 2000 && k < 4000, "This test may fail from time to time as it is based on sampling. k should be ~3333. k is " << k ); // Hopefully
}

/*
BOOST_AUTO_TEST_CASE( IO ) {
    const int S = 10, A = 8;

    std::string inputExpFilename     = "./data/experience.txt";
    std::string inputModelFilename   = "./data/model.txt";
    std::string outputFilename       = "./computedModule.txt";

    AIToolbox::MDP::SparseExperience exp(S,A);
    {
        std::ifstream inputExpFile(inputExpFilename);

        if ( !inputExpFile ) BOOST_FAIL("Data to perform test could not be loaded: " + inputExpFilename);
        BOOST_CHECK( inputExpFile >> exp );
    }

    AIToolbox::MDP::RLModel<decltype(exp)>  model(exp, 1.0, true);
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
