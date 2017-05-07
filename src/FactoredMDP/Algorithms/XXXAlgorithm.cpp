#include <AIToolbox/FactoredMDP/Algorithms/XXXAlgorithm.hpp>

#include <AIToolbox/FactoredMDP/Utils.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/Utils/UCVE.hpp>

#include <iostream>

namespace AIToolbox {
    namespace FactoredMDP {
        XXXAlgorithm::XXXAlgorithm(Action aa, const std::vector<std::pair<double, std::vector<size_t>>> & rangesAndDependencies) :
                A(std::move(aa)), timestep_(0),
                averages_(A), logA_(0.0)
        {
            // Compute log(|A|) without needing to compute |A| which may be too
            // big. We'll use it later to obtain log(t |A|)
            for (const auto a : A)
                logA_ += std::log(a);

            // Build single rules for each dependency group.
            // This allows us to allocate the rules_ only once, and to just
            // update their values at each timestep.
            for (const auto & dependency : rangesAndDependencies) {
                PartialFactorsEnumerator enumerator(A, dependency.second);
                while (enumerator.isValid()) {
                    const auto & pAction = *enumerator;

                    averages_.emplace(pAction, Average{ 0.0, 0, dependency.first * dependency.first });
                    rules_.emplace_back(pAction, UCVE::V());

                    enumerator.advance();
                }
            }
        }

        Action XXXAlgorithm::stepUpdateQ(const Action & a, const Rewards & rew) {
            // auto printaction = [](Action y){
            //     std::cout << "[";
            //     for (auto yy : y) std::cout << yy << ", ";
            //     std::cout << "]";
            // };
            // std::cout << "Input: ";
            // printaction(a);
            // std::cout << '\n';


            // std::cout << "Updating averages\n";

            // Update all averages with what we've learned this step.  Note: We
            // know that the factors are going to be in the correct order when
            // looping here since we are looping in the same order in which we
            // have inserted them into the averages_ container, and filter
            // returns sorted lists. So we can correctly match the rewards with
            // the agent groups!
            size_t i = 0;
            auto filtered = averages_.filter(a);
            for (auto & avg : filtered)
                avg.value += (rew[i++] - avg.value) / (++avg.count);

            // std::cout << "Building vectors\n";

            // Build the vectors to pass to UCVE
            for (size_t i = 0; i < averages_.size(); ++i) {
                const double count = averages_[i].count ? averages_[i].count : 0.00001;
                std::get<1>(rules_[i])[0] = averages_[i].value;
                std::get<1>(rules_[i])[1] = averages_[i].rangeSquared / count;
            }

            // for (const auto & v : ucveVectors) {
            //     std::cout << "PA:[";
            //     for (size_t i = 0; i < std::get<0>(v).first.size(); ++i)
            //         std::cout << std::get<0>(v).first[i] << ", " << std::get<0>(v).second[i] << " | ";
            //     std::cout << "] ==> " << std::get<1>(v).transpose() << '\n';
            // }

            // Update the timestep, and finish computing log(t |A|) for this
            // timestep.
            ++timestep_;
            const auto logtA = logA_ + std::log(timestep_);

            // std::cout << logtA << "\n";

            // Create and run UCVE
            UCVE ucve(A, logtA);
            auto a_v = ucve(rules_);

            // std::cout << "Result: ";
            // printaction(toFactors(A.size(), std::get<0>(vcs[0])));
            // std::cout << std::get<1>(vcs[0]).transpose() << '\n';

            // std::cout << "Returned vectors: " << vcs.size() << "\n";

            // We pick the first out since they should all be equally good, and
            // convert it to a normal action.
            return toFactors(A.size(), std::get<0>(a_v));
        }

        FactoredContainer<QFunctionRule> XXXAlgorithm::getQFunctionRules() const {
            FactoredContainer<QFunctionRule>::ItemsContainer container;

            for (size_t i = 0; i < averages_.size(); ++i)
                container.emplace_back(PartialState{}, std::get<0>(rules_[i]), averages_[i].value);

            return FactoredContainer<QFunctionRule>(averages_.getTrie(), std::move(container));
        }

        unsigned XXXAlgorithm::getTimestep() const { return timestep_; }
        void XXXAlgorithm::setTimestep(unsigned t) { timestep_ = t; }
    }
}
