#include <AIToolbox/Factored/MDP/CooperativeMaximumLikelihoodModel.hpp>

#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeMaximumLikelihoodModel::CooperativeMaximumLikelihoodModel(const CooperativeExperience & exp, const double discount, const bool toSync)
            : experience_(exp), discount_(discount), transitions_({experience_.getGraph(), {}})
    {
        const auto & S = experience_.getS();
        auto & tProbs = transitions_.transitions;

        tProbs.reserve(S.size());
        rewards_.reserve(S.size());

        for (size_t i = 0; i < S.size(); ++i) {
            const auto d1 = experience_.getGraph().getSize(i);
            const auto d2 = S[i];

            tProbs.emplace_back(d1, d2);
            rewards_.emplace_back(d1);

            tProbs.back().setZero();
            tProbs.back().col(0).fill(1.0);

            rewards_.back().setZero();
        }
        if (toSync) sync();
    }

    void CooperativeMaximumLikelihoodModel::sync() {
        const auto & S = experience_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            for (size_t j = 0; j < getGraph().getSize(i); ++j) {
                syncRow(i, j);
            }
        }
    }

    void CooperativeMaximumLikelihoodModel::sync(const State & s, const Action & a) {
        const auto & S = experience_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = experience_.getGraph().getId(i, s, a);

            syncRow(i, j);
        }
    }

    void CooperativeMaximumLikelihoodModel::sync(const CooperativeExperience::Indeces & indeces) {
        const auto & S = experience_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = indeces[i];

            syncRow(i, j);
        }
    }

    void CooperativeMaximumLikelihoodModel::syncRow(size_t i, size_t j) {
        const auto & S = experience_.getS();
        const auto & vtable  = experience_.getVisitsTable();
        const auto & rmatrix = experience_.getRewardMatrix();
        auto & tProbs = transitions_.transitions;

        const auto totalVisits = vtable[i](j, S[i]);
        if (totalVisits == 0) return;

        tProbs[i].row(j) = vtable[i].row(j).head(S[i]).cast<double>() / totalVisits;
        rewards_[i][j] = rmatrix[i][j];
    }

    std::tuple<State, double> CooperativeMaximumLikelihoodModel::sampleSR(const State & s, const Action & a) const {
        State s1(s.size());
        const double reward = sampleSR(s, a, &s1);

        return std::make_tuple(s1, reward);
    }

    std::tuple<State, Rewards> CooperativeMaximumLikelihoodModel::sampleSRs(const State & s, const Action & a) const {
        const auto & S = experience_.getS();

        State s1(S.size());
        Rewards rs(S.size());

        sampleSRs(s, a, &s1, &rs);

        return std::make_tuple(s1, rs);
    }

    double CooperativeMaximumLikelihoodModel::sampleSR(const State & s, const Action & a, State * s1p) const {
        assert(s1p);

        const auto & S = experience_.getS();
        auto & tProbs = transitions_.transitions;

        State & s1 = *s1p;

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = experience_.getGraph().getId(i, s, a);

            s1[i] = sampleProbability(S[i], tProbs[i].row(j), rand_);
        }

        return getExpectedReward(s, a, s1);
    }

    void CooperativeMaximumLikelihoodModel::sampleSRs(const State & s, const Action & a, State * s1p, Rewards * rews) const {
        assert(s1p);
        assert(rews);

        const auto & S = experience_.getS();
        auto & tProbs = transitions_.transitions;

        State & s1 = *s1p;

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = experience_.getGraph().getId(i, s, a);

            s1[i] = sampleProbability(S[i], tProbs[i].row(j), rand_);
        }

        getExpectedRewards(s, a, s1, rews);
    }

    double CooperativeMaximumLikelihoodModel::getTransitionProbability(const State & s, const Action & a, const State & s1) const {
        return transitions_.getTransitionProbability(s, a, s1);
    }

    double CooperativeMaximumLikelihoodModel::getExpectedReward(const State & s, const Action & a, const State &) const {
        const auto & S = experience_.getS();

        double retval = 0.0;
        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = experience_.getGraph().getId(i, s, a);

            retval += rewards_[i][j];
        }

        return retval;
    }

    Rewards CooperativeMaximumLikelihoodModel::getExpectedRewards(const State & s, const Action & a, const State & s1) const {
        const auto & S = experience_.getS();

        Rewards rews(S.size());

        getExpectedRewards(s, a, s1, &rews);

        return rews;
    }

    void CooperativeMaximumLikelihoodModel::getExpectedRewards(const State & s, const Action & a, const State &, Rewards * rewsp) const {
        assert(rewsp);

        const auto & S = experience_.getS();

        auto & rews = *rewsp;

        for (size_t i = 0; i < S.size(); ++i) {
            const auto j = experience_.getGraph().getId(i, s, a);

            rews[i] = rewards_[i][j];
        }
    }

    void CooperativeMaximumLikelihoodModel::setDiscount(const double d) { discount_ = d; }
    double CooperativeMaximumLikelihoodModel::getDiscount() const { return discount_; }

    const State & CooperativeMaximumLikelihoodModel::getS() const { return experience_.getS(); }
    const Action & CooperativeMaximumLikelihoodModel::getA() const { return experience_.getA(); }
    const CooperativeExperience & CooperativeMaximumLikelihoodModel::getExperience() const { return experience_; }
    const CooperativeMaximumLikelihoodModel::TransitionMatrix & CooperativeMaximumLikelihoodModel::getTransitionFunction() const { return transitions_; }
    const CooperativeMaximumLikelihoodModel::RewardMatrix & CooperativeMaximumLikelihoodModel::getRewardFunction() const { return rewards_; }
    const DDNGraph & CooperativeMaximumLikelihoodModel::getGraph() const { return experience_.getGraph(); }
}
