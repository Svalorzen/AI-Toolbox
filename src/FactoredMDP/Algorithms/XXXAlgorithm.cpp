#include <AIToolbox/FactoredMDP/Algorithms/XXXAlgorithm.hpp>

#include <AIToolbox/FactoredMDP/Utils.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/Utils/UCVE.hpp>

#include <iostream>

namespace AIToolbox {
    namespace FactoredMDP {
        XXXAlgorithm::XXXAlgorithm(Action a, const std::vector<std::pair<double, std::vector<size_t>>> & dependenciesAndRanges) :
                A(a), timestep_(0),
                graph_(A.size()), logA_(0.0)
        {
            // Compute log(|A|) without needing to compute |A| which may be too
            // big. We'll use it later to obtain log(t |A|)
            for (const auto a : A)
                logA_ += std::log(a);

            // std::cout << "loga: " << logA_ << '\n';

            // Build graph
            for (const auto & dependency : dependenciesAndRanges) {
                auto it = graph_.getFactor(dependency.second);

                // We already have this one.
                if (it->getData().averages.size() != 0) continue;

                it->getData().rangeSquared = dependency.first * dependency.first;
                it->getData().averages.resize(factorSpacePartial(dependency.second, A));
                // We initialize all counts to 1 to avoid infinities. This
                // helps at the start to explore actions in a more efficient
                // manner.
                for (auto & avg : it->getData().averages)
                    avg.count = 1;

                // std::cout << "rsq: " << it->getData().rangeSquared <<
                //              " -- size: " << it->getData().averages.size() << '\n';
            }
        }

        Action XXXAlgorithm::stepUpdateQ(const Action & a, const Rewards & rew) {
            assert((size_t)rew.size() == graph_.getFactors().size());

            // auto printaction = [](Action y){
            //     std::cout << "[";
            //     for (auto yy : y) std::cout << yy << ", ";
            //     std::cout << "]";
            // };
            // std::cout << "Input: ";
            // printaction(a);
            // std::cout << '\n';

            const auto begin = graph_.factorsBegin();
            const auto end   = graph_.factorsEnd();

            // std::cout << "Updating averages\n";

            // Update all averages with what we've learned this step.
            // Note: We know that the factors are going to be in the correct
            // order when looping here since we are looping in the same order
            // in which we have inserted them into the graph. So we can
            // correctly match the rewards with the agent groups!
            size_t i = 0;
            for (auto it = begin; it != end; ++it) {
                const auto & agents = graph_.getNeighbors(it);
                // Get previous data
                auto & avg = it->getData().averages[toIndexPartial(agents, A, a)];

                avg.value += (rew[i++] - avg.value) / (++avg.count);
            }

            // std::cout << "Building vectors\n";

            // Build the vectors to pass to UCVE
            UCVE::Entries ucveVectors;
            for (auto it = begin; it != end; ++it) {
                const auto & agents = graph_.getNeighbors(it);

                // std::cout << "Working for "<<agents.size()<< "agents\n";

                PartialFactorsEnumerator enumerator(A, agents);
                while (enumerator.isValid()) {
                    const auto & pAction = *enumerator;
                    // Get the average structure for this action
                    const auto & avg = it->getData().averages[toIndexPartial(A, pAction)];
                    // Create new vector
                    ucveVectors.emplace_back(
                        pAction,
                        UCVE::V{
                            avg.value,
                            it->getData().rangeSquared / avg.count
                        }
                    );
                    enumerator.advance();
                }
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
            auto logtA = logA_ + std::log(timestep_);

            // std::cout << logtA << "\n";

            // Create and run UCVE
            UCVE ucve(A, logtA);
            auto a_v = ucve(ucveVectors);

            // std::cout << "Result: ";
            // printaction(toFactors(A.size(), std::get<0>(vcs[0])));
            // std::cout << std::get<1>(vcs[0]).transpose() << '\n';

            // std::cout << "Returned vectors: " << vcs.size() << "\n";

            // We pick the first out since they should all be equally good, and
            // convert it to a normal action.
            return toFactors(A.size(), std::get<0>(a_v));
        }

        unsigned XXXAlgorithm::getTimestep() const { return timestep_; }
        void XXXAlgorithm::setTimestep(unsigned t) { timestep_ = t; }

        // TODO: Convert vectors to QFunctionRules (skipping exploration) to allow using other policies.
    }
}
