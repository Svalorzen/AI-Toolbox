#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>

#include <AIToolbox/Seeder.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeModel::CooperativeModel(DDNGraph graph, DDN::TransitionMatrix transitions, FactoredMatrix2D rewards, const double discount) :
            discount_(discount),
            graph_(std::move(graph)),
            transitions_({graph_, std::move(transitions)}), rewards_(std::move(rewards)),
            rand_(Seeder::getSeed())
    {
        // Now we validate both the transition function and the rewards.
        // The DDN graph we can already trust since it's a class and not a
        // struct, but we can still check that all the nodes have been pushed.
        const auto & S = graph_.getS();
        const auto & A = graph_.getA();
        if (S.size() == 0)
            throw std::invalid_argument("Input DDN has empty state space in its DDNGraph");
        if (A.size() == 0)
            throw std::invalid_argument("Input DDN has empty action space in its DDNGraph");
        if (graph_.getParentSets().size() != S.size())
            throw std::invalid_argument("Input DDN has an incorrect number of nodes in its DDNGraph!");

        if (transitions_.transitions.size() != S.size())
            throw std::invalid_argument("Input transition function has an incorrect number of transition nodes!");

        for (size_t i = 0; i < S.size(); ++i) {
            if (static_cast<size_t>(transitions_.transitions[i].rows()) != graph_.getSize(i)) {
                throw std::invalid_argument("Input transition matrix " + std::to_string(i) + " contains an incorrect number of rows!");
            }
            if (static_cast<size_t>(transitions_.transitions[i].cols()) != graph_.getS()[i]) {
                throw std::invalid_argument("Input transition matrix " + std::to_string(i) + " contains an incorrect number of columns!");
            }

            // Check each row is a probability.
            for (size_t j = 0; j < graph_.getSize(i); ++j)
                if (!isProbability(S[i], transitions_.transitions[i].row(j)))
                    throw std::invalid_argument("Input transition matrix " + std::to_string(i) + " contains invalid probabilities at row " + std::to_string(j) + '!');
        }

        for (size_t i = 0; i < rewards_.bases.size(); ++i) {
            const auto & r = rewards_.bases[i];

            auto [error, id] = checkTag(A, r.actionTag);
            switch (error) {
                case TagErrors::NoElements:
                    throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains an action tag with no elements!");
                case TagErrors::TooManyElements:
                    throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains an action tag with too many elements!");
                case TagErrors::IdTooHigh:
                    throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains an action tag with action IDs too high for the action space!");
                case TagErrors::NotSorted:
                    throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains an action tag that is not sorted!");
                case TagErrors::Duplicates:
                    throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains an action tag with duplicates!");
                default:;
            }
            std::tie(error, id) = checkTag(S, r.tag);
            switch (error) {
                case TagErrors::NoElements:
                    throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains a state tag with no elements!");
                case TagErrors::TooManyElements:
                    throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains a state tag with too many elements!");
                case TagErrors::IdTooHigh:
                    throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains a state tag with state IDs too high for the state space!");
                case TagErrors::NotSorted:
                    throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains a state tag that is not sorted!");
                case TagErrors::Duplicates:
                    throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains a state tag with duplicates!");
                default:;
            }

            // Check size of matrix is correct
            if (r.values.cols() != static_cast<long>(factorSpacePartial(r.actionTag, A)))
                throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains an incorrect number of columns!");

            if (r.values.rows() != static_cast<long>(factorSpacePartial(r.tag, S)))
                throw std::invalid_argument("Input reward function base " + std::to_string(i) + " contains an incorrect number of rows!");
        }
    }

    CooperativeModel::CooperativeModel(const CooperativeModel & other) :
            discount_(other.discount_),
            graph_(other.graph_),
            transitions_({graph_, other.transitions_.transitions}), rewards_(other.rewards_),
            rand_(other.rand_)
    {}

    std::tuple<State, double> CooperativeModel::sampleSR(const State & s, const Action & a) const {
        const auto & S = graph_.getS();

        State s1(S.size());
        const double reward = sampleSR(s, a, &s1);

        return std::make_tuple(s1, reward);
    }

    double CooperativeModel::sampleSR(const State & s, const Action & a, State * s1p) const {
        const auto & tProbs = transitions_.transitions;
        const auto & S = graph_.getS();

        State & s1 = *s1p;

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = graph_.getId(i, s, a);

            s1[i] = sampleProbability(S[i], tProbs[i].row(j), rand_);
        }

        return rewards_.getValue(S, graph_.getA(), s, a);
    }

    std::tuple<State, Rewards> CooperativeModel::sampleSRs(const State & s, const Action & a) const {
        const auto & S = graph_.getS();

        std::tuple<State, Rewards> retval;
        auto & [s1, rews] = retval;

        s1.resize(S.size());
        rews.resize(rewards_.bases.size());

        sampleSRs(s, a, &s1, &rews);

        return retval;
    }

    void CooperativeModel::sampleSRs(const State & s, const Action & a, State * s1p, Rewards * rp) const {
        assert(s1p);
        assert(rp);

        auto & s1 = *s1p;
        auto & rews = *rp;

        const auto & tProbs = transitions_.transitions;
        const auto & S = graph_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = graph_.getId(i, s, a);

            s1[i] = sampleProbability(S[i], tProbs[i].row(j), rand_);
        }

        for (size_t i = 0; i < rewards_.bases.size(); ++i) {
            const auto & e = rewards_.bases[i];
            const auto fid = toIndexPartial(e.tag, S, s);
            const auto aid = toIndexPartial(e.actionTag, graph_.getA(), a);

            rews[i] = e.values(fid, aid);
        }
    }

    double CooperativeModel::getTransitionProbability(const State & s, const Action & a, const State & s1) const {
        return transitions_.getTransitionProbability(s, a, s1);
    }

    double CooperativeModel::getExpectedReward(const State & s, const Action & a, const State &) const {
        return rewards_.getValue(graph_.getS(), graph_.getA(), s, a);
    }

    const State & CooperativeModel::getS() const { return graph_.getS(); }
    const Action & CooperativeModel::getA() const { return graph_.getA(); }
    double CooperativeModel::getDiscount() const { return discount_; }
    const DDN & CooperativeModel::getTransitionFunction() const { return transitions_; }
    const FactoredMatrix2D & CooperativeModel::getRewardFunction() const { return rewards_; }
    const DDNGraph & CooperativeModel::getGraph() const { return graph_; }
}
