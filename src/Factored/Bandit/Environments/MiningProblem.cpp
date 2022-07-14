#include <AIToolbox/Factored/Bandit/Environments/MiningProblem.hpp>

#include <random>
#include <iostream>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/GraphUtils.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::Bandit {
    namespace {
        double rewFun(double productivity, size_t totalWorkers) {
            assert(totalWorkers > 0);
            return productivity * std::pow(1.03, totalWorkers);
        };
    }

    MiningBandit::MiningBandit(Action aa, std::vector<unsigned> workersPerVillage, std::vector<double> pPerMine, bool normalizeToOne) :
        A(std::move(aa)), workersPerVillage_(std::move(workersPerVillage)), productivityPerMine_(std::move(pPerMine)),
        normalizeToOne_(normalizeToOne),
        villagesPerMine_(productivityPerMine_.size()),
        helper_(productivityPerMine_.size())
    {
        assert(workersPerVillage_.size() == A.size());
        // Last village should have 4 possible mines.
        assert(productivityPerMine_.size() == A.size() + 3);
        assert(A.back() == 4);

        for (size_t v = 0; v < A.size(); ++v)
            for (size_t m = v; m < v + A[v]; ++m)
                villagesPerMine_[m].push_back(v);

        // Compute optimal action in advance so we can:
        // - Normalize rewards so that the optimal action's expected reward is 1.0
        // - Provide an accurate regret
        //
        // To do so we solve, with VE, the exact problem knowing the expected
        // rewards for each action (we have no random sampling here).
        auto rules = getDeterministicRules();

        VariableElimination ve;
        auto g = MakeGraph<VariableElimination>()(rules, A);
        UpdateGraph<VariableElimination>()(g, rules, A);
        std::tie(optimal_, rewardNorm_) = ve(A, g);

        // If we don't normalize the best reward to 1, then we simply normalize
        // values so that productivities can never exceed 1 (since we use
        // Bernoullis to sample).
        if (!normalizeToOne_) {
            rewardNorm_ = 1.0;

            // For each mine, determine the maximum amount of workers that can go to it.
            for (size_t m = 0; m < productivityPerMine_.size(); ++m) {
                unsigned totalMiners = 0;
                for (size_t v : villagesPerMine_[m])
                    totalMiners += workersPerVillage_[v];

                const double maxRew = rewFun(productivityPerMine_[m], totalMiners);
                rewardNorm_ = std::max(rewardNorm_, maxRew);
            }
        }
    }

    const Rewards & MiningBandit::sampleR(const Action & a) const {
        computeProbabilities(a);

        for (size_t m = 0; m < productivityPerMine_.size(); ++m) {
            std::bernoulli_distribution roll(helper_[m]);
            helper_[m] = static_cast<double>(roll(rand_));
        }

        return helper_;
    };

    double MiningBandit::getRegret(const Action & a) const {
        // Special case for optimal action to avoid returning floating point
        // fluff close to zero.
        if (a == optimal_) return 0.0;

        computeProbabilities(a);

        return 1.0 - helper_.sum();
    }

    void MiningBandit::computeProbabilities(const Action & a) const {
        assert(a.size() == A.size());

        // Count workers per mine
        helper_.fill(0.0);
        for (size_t v = 0; v < a.size(); ++v)
            helper_[v + a[v]] += workersPerVillage_[v];

        for (size_t m = 0; m < productivityPerMine_.size(); ++m) {
            // If there are no workers, there is zero reward.
            if (helper_[m] == 0.0) continue;

            // Otherwise, apply mine formula (productivity * 1.03 ^ miners)
            // and normalize the resulting probability.
            helper_[m] = rewFun(productivityPerMine_[m], helper_[m]) / rewardNorm_;
        }
    }

    std::vector<QFunctionRule> MiningBandit::getDeterministicRules() const {
        // Here we simply aggregate, for each mine, the true values it would
        // return for any given local joint action of its attached villages.
        std::vector<QFunctionRule> rules;
        for (size_t m = 0; m < villagesPerMine_.size(); ++m) {
            const auto & mineVillages = villagesPerMine_[m];

            PartialFactorsEnumerator enumerator(A, mineVillages);
            while (enumerator.isValid()) {
                const auto & villageAction = enumerator->second;

                unsigned totalMiners = 0;
                for (size_t v = 0; v < mineVillages.size(); ++v)
                    if (mineVillages[v] + villageAction[v] == m)
                        totalMiners += workersPerVillage_[mineVillages[v]];

                if (totalMiners > 0) {
                    const double v = rewFun(productivityPerMine_[m], totalMiners);
                    rules.emplace_back(*enumerator, v);
                }

                enumerator.advance();
            }
        }
        return rules;
    }

    const std::vector<PartialKeys> & MiningBandit::getGroups() const { return villagesPerMine_; }
    const Action & MiningBandit::getA() const { return A; }
    const Action & MiningBandit::getOptimalAction() const { return optimal_; }
    double MiningBandit::getNormalizationConstant() const { return rewardNorm_; }

    std::tuple<Action, std::vector<unsigned>, std::vector<double>> makeMiningParameters(unsigned seed) {
        RandomEngine rand(seed);

        std::uniform_int_distribution<size_t> villages(5, 15);
        std::uniform_int_distribution<unsigned> workersPerVillage(1, 5);
        std::uniform_int_distribution<size_t> minesPerVillage(2, 4);
        std::uniform_real_distribution<double> mineP(0, 0.5);

        // Generate villages and attached mines
        const auto villagesNum = villages(rand);
        const auto minesNum = villagesNum + 3;

        Action A(villagesNum);
        std::vector<unsigned> workers(villagesNum);

        // For each village, determine to how many mines it can go, and how
        // many workers it has.
        for (size_t n = 0; n < villagesNum; ++n) {
            workers[n] = workersPerVillage(rand);
            A[n]       = minesPerVillage(rand);

            // Last village always has 4 mines
            if (n == villagesNum - 1) A[n] = 4;
        }

        // Find out which villages are attached to each mine

        // Compute probabilities for each mine
        auto minePs = std::vector<double>(minesNum);
        for (size_t m = 0; m < minesNum; ++m)
            minePs[m] = mineP(rand);

        return {A, workers, minePs};
    }
}
