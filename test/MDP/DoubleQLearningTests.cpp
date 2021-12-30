#define BOOST_TEST_MODULE MDP_DoubleQLearning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "GlobalFixtures.hpp"

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/Algorithms/DoubleQLearning.hpp>
#include <AIToolbox/MDP/Algorithms/QLearning.hpp>
#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/MDP/Policies/EpsilonPolicy.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

#include <iostream>

class Roulette {
    public:
        Roulette() :
            odds_({
                    std::bernoulli_distribution{1.0/(37.0 + 1.0)},              // 37 to 1
                    std::bernoulli_distribution{1.0/(18.0 + 1.0)},              // 18 to 1
                    std::bernoulli_distribution{1.0/(11.0 + 2.0 / 3.0 + 1.0)},  // 11 2/3 to 1
                    std::bernoulli_distribution{1.0/( 8.0 + 1.0 / 2.0 + 1.0)},  // 8 1/2 to 1
                    std::bernoulli_distribution{1.0/( 6.0 + 3.0 / 5.0 + 1.0)},  // 6 3/5 to 1
                    std::bernoulli_distribution{1.0/( 5.0 + 1.0 / 3.0 + 1.0)},  // 5 1/3 to 1
                    std::bernoulli_distribution{1.0/( 2.0 + 1.0 / 6.0 + 1.0)},  // 2 1/6 to 1
                    std::bernoulli_distribution{1.0/( 1.0 + 1.0 / 9.0 + 1.0)}   // 1 1/9 to 1
            }),
            payouts_({35.0, 17.0, 11.0, 8.0, 6.0, 5.0, 2.0, 1.0}),
            // Numbers between parentheses are added only to make the total
            // number of betting actions equal to 170 as in the paper; I don't
            // know what else they could be.
            numActions_({
                38,     // Single numbers
                3 + 11 * 3 + 2 * 12 + (5), // 0-00, 0-1, 00-3; Adjacent pairs
                12 + 3 + (4), // Row of 3 numbers; 0-1-2, 0-00-2, 00-2-3
                2 * 11 + (2), // Block of 4 numbers
                1,      // Top line (0-00-1-2-3)
                11,     // Six line
                6,      // Column 1,2,3; Dozen 1,2,3
                6       // Odd/Even, Red/Black, 1-18/19-36
            })
        {
            assert([this](){
                size_t tot = 0;
                for (auto a : numActions_)
                    tot += a;
                return tot == 170;
            }());
        }

        size_t getS() const { return 2; }
        size_t getA() const { return 171; }
        double getDiscount() const { return 0.95; }

        std::tuple<size_t, double> sampleSR(size_t s, size_t a) const {
            if (s == 1 || a == 170) return {1, 0.0};

            double r = -1.0;
            for (size_t i = 0; i < odds_.size(); ++i) {
                if (a >= numActions_[i]) {
                    a -= numActions_[i];
                    continue;
                }
                if (odds_[i](rand_)) r += payouts_[i];
                break;
            }

            return {0, r};
        }
        bool isTerminal(size_t s) const { return s; }

    private:
        mutable std::array<std::bernoulli_distribution, 8> odds_;
        std::array<double, 8> payouts_;
        std::array<unsigned, 8> numActions_;

        mutable AIToolbox::RandomEngine rand_;
};

BOOST_AUTO_TEST_CASE( roulette ) {
    using namespace AIToolbox::MDP;

    Roulette model;
    Experience exp(model.getS(), model.getA());

    DoubleQLearning solver(model, 0.5);
    // QLearning solver(model, 0.5);

    QGreedyPolicy gPolicy(solver.getQFunction());
    EpsilonPolicy ePolicy(gPolicy, 0.1);

    size_t start = model.getS() - 2;

    size_t s, a;
    for ( int episode = 0; episode < 100; ++episode ) {
        s = start;
        for ( int i = 0; i < 10000; ++i ) {
            a = ePolicy.sampleAction( s );
            const auto [s1, rew] = model.sampleSR( s, a );

            const double lr = std::max(1.0, std::pow(exp.getVisitsSum(s, a), 0.8));

            solver.setLearningRate(1.0 / lr);
            solver.stepUpdateQ(s, a, s1, rew);

            exp.record(s, a, s1, rew);

            if (model.isTerminal(s1)) break;
            s = s1;
        }
    }

    BOOST_CHECK(solver.getQFunction().row(0).maxCoeff() <= 0.0);
    // We leave some space for random underestimation
    BOOST_CHECK(solver.getQFunction().row(0).minCoeff() >= -15.0);

    // Everything else should be zero
    BOOST_CHECK_EQUAL(solver.getQFunction().row(1).maxCoeff(), 0.0);
    BOOST_CHECK_EQUAL(solver.getQFunction().row(1).minCoeff(), 0.0);
}

BOOST_AUTO_TEST_CASE( exceptions ) {
    using namespace AIToolbox::MDP;

    BOOST_CHECK_EXCEPTION(DoubleQLearning(1,1,0.0,0.5),   std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(DoubleQLearning(1,1,-10.0,0.5), std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(DoubleQLearning(1,1,3.0,0.5),   std::invalid_argument, [](const std::invalid_argument &){return true;});

    BOOST_CHECK_EXCEPTION(DoubleQLearning(1,1,0.3,0.0),   std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(DoubleQLearning(1,1,0.3,-0.5),  std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(DoubleQLearning(1,1,0.3,1.1),   std::invalid_argument, [](const std::invalid_argument &){return true;});
}

