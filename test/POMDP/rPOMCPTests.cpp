#define BOOST_TEST_MODULE POMDP_rPOMCP
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Algorithms/rPOMCP.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/POMDP/Utils.hpp>

#include <AIToolbox/Utils/Probability.hpp>

/**
 * @brief The model for rPOMCP tests
 *
 * This model has been specially designed to require different answers
 * depending on whether rPOMCP is running in max-belief or entropy mode.
 *
 * Note how the model doesn't really give reward, since rPOMCP uses its own
 * belief-based reward function.
 *
 * The idea is to have a state which represents two cells, like this:
 *
 * [0|1] [2|3]  <-- state
 *   ^     ^
 *   0     1    <-- action
 *
 * With a low probability, the action chosen by the agent will influence the
 * next state by moving the probability from one set of states to the other.
 *
 * There is no way however to influence which state of the set is chosen -
 * there's a small probability it won't change to avoid having a completely
 * random state.
 *
 * In the tests we start from a belief like {0.6, 0.0, 0.2, 0.2}. The
 * max-belief solution is to chose the first set, since selecting the second
 * would lower the max-belief by a large amount.
 *
 * On the other hand, the entropy solution is to select the second set, which
 * prevents the state from going to state 1, thus resulting in a lower entropy.
 *
 * This test is *somewhat* brittle; if we increase the horizon to say 7 for the
 * entropy rPOMCP it sometimes gets the wrong answer. But again, it could be
 * because the probabilities/rewards there are really close, so it could be
 * possible that it's just a sampling error - I haven't done the math for it.
 */
class Model {
    public:
        Model() : dist_(0.0, 1.0) {}

        size_t getS() const { return 4; }
        size_t getA() const { return 2; }
        double getDiscount() const { return 0.9; }
        std::tuple<size_t, double> sampleSR(const size_t s, const size_t a) const {
            if (dist_(rand_) > 0.17)
                return std::make_tuple(s, a);

            auto p = dist_(rand_);
            auto swap = dist_(rand_) < 0.5;

            auto ss = s;

            if (a == 0 && (s == 0 || s == 1)) {
                if (p > 0.3)
                    ss = swap;
            } else if (a == 1 && (s == 2 || s == 3)) {
                if (p > 0.3)
                    ss = swap + 2;
            } else {
                if (s > 1)
                    ss = swap;
                else
                    ss = swap + 2;
            }

            return std::make_tuple(ss, 0.0);
        }

        std::tuple<size_t,size_t,double> sampleSOR(const size_t s, const size_t a) const {
            const auto s1_r = sampleSR(s, a);

            return std::make_tuple(std::get<0>(s1_r), 0, std::get<1>(s1_r));
        }
        bool isTerminal(size_t) const { return false; }

    private:
        // We need this because we don't know if our parent already has one,
        // and we wouldn't know how to access it!
        mutable AIToolbox::RandomEngine rand_;
        mutable std::uniform_real_distribution<double> dist_;
};

BOOST_AUTO_TEST_CASE( entropy ) {
    using namespace AIToolbox;

    Model model;

    Matrix2D beliefs(2, 4);
    beliefs << 0.2, 0.2, 0.0, 0.6,
               0.6, 0.0, 0.2, 0.2;

    std::vector<size_t> solutions{0, 1};

    for ( auto i = 0; i < beliefs.rows(); ++i ) {
        POMDP::rPOMCP<decltype(model), true> solver(model, 1000, 50000, 200.0);

        BOOST_CHECK_EQUAL(solver.sampleAction(beliefs.row(i), 2), solutions[i]);
    }
}

BOOST_AUTO_TEST_CASE( max_belief ) {
    using namespace AIToolbox;

    Model model;

    Matrix2D beliefs(2, 4);
    beliefs << 0.2, 0.2, 0.0, 0.6,
               0.6, 0.0, 0.2, 0.2;

    std::vector<size_t> solutions{1, 0};

    for ( auto i = 0; i < beliefs.rows(); ++i ) {
        POMDP::rPOMCP<decltype(model), false> solver(model, 1000, 50000, 200.0);

        BOOST_CHECK_EQUAL(solver.sampleAction(beliefs.row(i), 2), solutions[i]);
    }
}
