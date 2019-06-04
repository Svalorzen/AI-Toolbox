#include <AIToolbox/Factored/MDP/CooperativeRLModel.hpp>

#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeRLModel::CooperativeRLModel(const CooperativeExperience & exp, const double discount, const bool toSync)
            : experience_(exp), discount_(discount)
    {
        const auto & S = experience_.getS();
        const auto & vnodes = experience_.getVisitTable();
        const auto & rnodes = experience_.getRewardMatrix();

        transitions_.nodes.resize(rnodes.size());
        rewards_.resize(rnodes.size());

        auto & tnodes = transitions_.nodes;

        for (size_t i = 0; i < rnodes.size(); ++i) {
            tnodes[i].actionTag = rnodes[i].actionTag;

            tnodes[i].nodes.resize(rnodes[i].nodes.size());
            rewards_[i].resize(rnodes[i].nodes.size());

            for (size_t j = 0; j < rnodes[i].nodes.size(); ++j) {
                tnodes[i].nodes[j].tag = rnodes[i].nodes[j].tag;

                tnodes[i].nodes[j].matrix.resize(
                    rnodes[i].nodes[j].matrix.rows(),
                    rnodes[i].nodes[j].matrix.cols() - 1 // vnodes also has the overall sum column
                );
                rewards_[i][j].resize(rnodes[i].nodes[j].matrix.rows());

                if (!toSync) {
                    tnodes[i].nodes[j].matrix.setZero();
                    tnodes[i].nodes[j].matrix.col(0).fill(1.0);

                    rewards_[i][j].setZero();
                } else {
                    for (int p = 0; p < tnodes[i].nodes[j].matrix.rows(); ++p) {
                        const double totalVisits = vnodes[i][j](p, S[i]);
                        if (totalVisits == 0) {
                            tnodes[i].nodes[j].matrix.row(p).setZero();
                            tnodes[i].nodes[j].matrix(p, 0) = 1.0;

                            rewards_[i][j][p] = 0.0;
                            continue;
                        }

                        tnodes[i].nodes[j].matrix.row(p) = vnodes[i][j].row(p).head(S[i]).cast<double>() / totalVisits;
                        rewards_[i][j][p] = rnodes[i].nodes[j].matrix(p, S[i]) / totalVisits;
                    }
                }
            }
        }
    }

    void CooperativeRLModel::sync() {
        const auto & S = experience_.getS();
        const auto & vnodes = experience_.getVisitTable();
        const auto & rnodes = experience_.getRewardMatrix();

        auto & tnodes = transitions_.nodes;

        for (size_t i = 0; i < rnodes.size(); ++i) {
            for (size_t j = 0; j < rnodes[i].nodes.size(); ++j) {
                for (int p = 0; p < tnodes[i].nodes[j].matrix.rows(); ++p) {
                    const double totalVisits = vnodes[i][j](p, S[i]);
                    if (totalVisits == 0) continue;

                    tnodes[i].nodes[j].matrix.row(p) = vnodes[i][j].row(p).head(S[i]).cast<double>() / totalVisits;
                    rewards_[i][j][p] = rnodes[i].nodes[j].matrix(p, S[i]) / totalVisits;
                }
            }
        }
    }

    void CooperativeRLModel::sync(const State & s, const Action & a) {
        const auto & vnodes = experience_.getVisitTable();
        const auto & rnodes = experience_.getRewardMatrix();

        auto & tnodes = transitions_.nodes;
        const auto & S = experience_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            const auto aId = toIndexPartial(tnodes[i].actionTag, getA(), a);

            const auto & node = tnodes[i].nodes[aId];
            const auto pId = toIndexPartial(node.tag, getS(), s);

            const double totalVisits = vnodes[i][aId](pId, S[i]);
            if (totalVisits == 0) continue;

            tnodes[i].nodes[aId].matrix.row(pId) = vnodes[i][aId].row(pId).head(S[i]).cast<double>() / totalVisits;

            rewards_[i][aId][pId] = rnodes[i].nodes[aId].matrix(pId, S[i]) / totalVisits;
        }
    }

    void CooperativeRLModel::sync(const CooperativeExperience::Indeces & indeces) {
        const auto & vnodes = experience_.getVisitTable();
        const auto & rnodes = experience_.getRewardMatrix();

        auto & tnodes = transitions_.nodes;
        const auto & S = experience_.getS();

        for (size_t i = 0; i < S.size(); ++i) {
            const auto [aId, pId] = indeces[i];

            const double totalVisits = vnodes[i][aId](pId, S[i]);
            if (totalVisits == 0) continue;

            tnodes[i].nodes[aId].matrix.row(pId) = vnodes[i][aId].row(pId).head(S[i]).cast<double>() / totalVisits;

            rewards_[i][aId][pId] = rnodes[i].nodes[aId].matrix(pId, S[i]) / totalVisits;
        }
    }

    std::tuple<State, double> CooperativeRLModel::sampleSR(const State & s, const Action & a) const {
        State s1(s.size());
        const double reward = sampleSR(s, a, &s1);

        return std::make_tuple(s1, reward);
    }

    std::tuple<State, Rewards> CooperativeRLModel::sampleSRs(const State & s, const Action & a) const {
        State s1(s.size());
        Rewards rs(s.size());

        sampleSRs(s, a, &s1, &rs);

        return std::make_tuple(s1, rs);
    }

    double CooperativeRLModel::sampleSR(const State & s, const Action & a, State * s1p) const {
        assert(s1p);

        const auto & tnodes = transitions_.nodes;
        State & s1 = *s1p;

        for (size_t i = 0; i < s.size(); ++i) {
            const auto actionId = toIndexPartial(tnodes[i].actionTag, getA(), a);

            const auto & node = tnodes[i].nodes[actionId];
            const auto parentId = toIndexPartial(node.tag, getS(), s);

            const size_t newS = sampleProbability(getS()[i], node.matrix.row(parentId), rand_);

            s1[i] = newS;
        }

        return getExpectedReward(s, a, s1);
    }

    void CooperativeRLModel::sampleSRs(const State & s, const Action & a, State * s1p, Rewards * rews) const {
        assert(s1p);
        assert(rews);

        const auto & tnodes = transitions_.nodes;
        State & s1 = *s1p;

        for (size_t i = 0; i < s.size(); ++i) {
            const auto actionId = toIndexPartial(tnodes[i].actionTag, getA(), a);

            const auto & node = tnodes[i].nodes[actionId];
            const auto parentId = toIndexPartial(node.tag, getS(), s);

            const size_t newS = sampleProbability(getS()[i], node.matrix.row(parentId), rand_);

            s1[i] = newS;
        }

        getExpectedRewards(s, a, s1, rews);
    }

    double CooperativeRLModel::getTransitionProbability(const State & s, const Action & a, const State & s1) const {
        return transitions_.getTransitionProbability(getS(), getA(), s, a, s1);
    }

    double CooperativeRLModel::getExpectedReward(const State & s, const Action & a, const State &) const {
        double retval = 0.0;

        for (size_t i = 0; i < transitions_.nodes.size(); ++i) {
            const auto actionId = toIndexPartial(transitions_[i].actionTag, getA(), a);

            const auto & node = transitions_[i].nodes[actionId];
            const auto parentId = toIndexPartial(node.tag, getS(), s);

            retval += rewards_[i][actionId][parentId];
        }

        return retval;
    }

    Rewards CooperativeRLModel::getExpectedRewards(const State & s, const Action & a, const State & s1) const {
        Rewards rews(transitions_.nodes.size());

        getExpectedRewards(s, a, s1, &rews);

        return rews;
    }

    void CooperativeRLModel::getExpectedRewards(const State & s, const Action & a, const State &, Rewards * rewsp) const {
        assert(rewsp);

        auto & rews = *rewsp;
        for (size_t i = 0; i < transitions_.nodes.size(); ++i) {
            const auto actionId = toIndexPartial(transitions_[i].actionTag, getA(), a);

            const auto & node = transitions_[i].nodes[actionId];
            const auto parentId = toIndexPartial(node.tag, getS(), s);

            rews[i] = rewards_[i][actionId][parentId];
        }
    }

    void CooperativeRLModel::setDiscount(const double d) { discount_ = d; }
    double CooperativeRLModel::getDiscount() const { return discount_; }

    const State & CooperativeRLModel::getS() const { return experience_.getS(); }
    const Action & CooperativeRLModel::getA() const { return experience_.getA(); }
    const CooperativeExperience & CooperativeRLModel::getExperience() const { return experience_; }
    const CooperativeRLModel::TransitionMatrix & CooperativeRLModel::getTransitionFunction() const { return transitions_; }
    const CooperativeRLModel::RewardMatrix & CooperativeRLModel::getRewardFunction() const { return rewards_; }
}
