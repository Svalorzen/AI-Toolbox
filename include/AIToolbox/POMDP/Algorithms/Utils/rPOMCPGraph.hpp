#ifndef AI_TOOLBOX_POMDP_rPOMCP_GRAPH_HEADER_FILE
#define AI_TOOLBOX_POMDP_rPOMCP_GRAPH_HEADER_FILE

#include <vector>
#include <unordered_map>

#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox::Impl::POMDP {
    struct EmptyStruct {};

    struct BeliefParticleEntropyAddon {
        double negativeEntropy = 0; ///< Estimated entropy deriving from this particle type
    };

    struct BeliefNodeNoEntropyAddon {
        size_t maxS_ = 0;           ///< This keeps track of the belief peak state for max of belief
    };
}

namespace AIToolbox::POMDP {
    template <bool UseEntropy>
    struct ActionNode;

    template <bool UseEntropy>
    using ActionNodes = std::vector<ActionNode<UseEntropy>>;

    template <bool UseEntropy>
    struct BeliefParticle : public std::conditional_t<UseEntropy, Impl::POMDP::BeliefParticleEntropyAddon, Impl::POMDP::EmptyStruct> {
        unsigned N = 0;             ///< Number of particles for this particular type (state)
    };

    // This is used to keep track of beliefs down in the tree. We use a map since
    // we do not need to sample from here, just to access fast and recompute the
    // entropy values.
    template <bool UseEntropy>
    using TrackBelief = std::unordered_map<
                            size_t,
                            BeliefParticle<UseEntropy>,
                            std::hash<size_t>,
                            std::equal_to<size_t>
                        >;

    /**
     * @brief This is a belief node of the rPOMCP tree.
     */
    template <bool UseEntropy>
    class BeliefNode : public std::conditional_t<UseEntropy, Impl::POMDP::EmptyStruct, Impl::POMDP::BeliefNodeNoEntropyAddon> {
        public:
            BeliefNode();

            /// This function updates the knowledge measure after adding a new belief particle.
            void updateBeliefAndKnowledge(size_t s);

            /// This function returns the current estimate for reward for this node.
            double getKnowledgeMeasure() const;

            unsigned N;          ///< Counter for number of times we went through this belief node.
            ActionNodes<UseEntropy> children;

            double V;            ///< Estimated value for this belief, taking into account future rewards/actions.
            double actionsV;     ///< Estimated value for the actions (could be mean, max, or other)
            size_t bestAction;   ///< Tracker of best available action in MAX-mode, to select node value.

        protected:
            /// This is a particle belief which is easy to update
            TrackBelief<UseEntropy> trackBelief_;
            /// Estimated entropy/max-belief for this node.
            double knowledgeMeasure_;
    };

    template <bool UseEntropy>
    using BeliefNodes = std::unordered_map<size_t, BeliefNode<UseEntropy>>;

    template <bool UseEntropy>
    struct ActionNode {
        BeliefNodes<UseEntropy> children;
        double V       = 0.0; ///< Tracks the value of the action, as a weighted average of the values of the next step beliefNodes.
        unsigned N     = 0;   ///< Times this action has been performed
    };

    // This is used to sample at the top of the tree. It is a vector containing a
    // state-count pair for each particle.
    using SampleBelief = std::vector<std::pair<size_t, unsigned>>;

    /**
     * @brief This class is the root node of the rPOMCP graph.
     *
     * This converts the unordered belief map of an ordinary belief node into a
     * vector. This should speed up the sampling process considerably, since
     * the head node is the one that gets sampled the most.
     *
     * Note that for this reason this node does not use the trackBelief_ field.
     * It uses the sampleBelief_ instead.
     */
    template <bool UseEntropy>
    class HeadBeliefNode : public BeliefNode<UseEntropy> {
        public:
            /**
             * @brief Basic constructor.
             */
            HeadBeliefNode(size_t A, RandomEngine & rand);
            /**
             * @brief Constructor from belief.
             *
             * This constructor samples the input belief in order to create the
             * sample belief. We create `beliefSize` samples.
             *
             * We don't store the belief directly since the sampleBelief must
             * also be constructible from the particle beliefs of normal belief
             * nodes (when we use rPOMCP for multiple timesteps). So we use a
             * particle belief in both cases.
             */
            HeadBeliefNode(size_t A, size_t beliefSize, const AIToolbox::POMDP::Belief & b, RandomEngine & rand);
            /**
             * @brief Constructor from BeliefNode.
             *
             * This constructor converts the input BeliefNode into a new head
             * node. It converts the track belief of the node into our sample
             * belief.
             */
            HeadBeliefNode(size_t A, BeliefNode<UseEntropy> && bn, RandomEngine & rand);

            bool isSampleBeliefEmpty() const;     ///< Whether we have no particles in the sampling belief.
            size_t sampleBelief() const;          ///< Samples the internal sampling belief.
            size_t getMostCommonParticle() const; ///< Useful if the agents wants a guess of what the current state is.

        private:
            RandomEngine * rand_;                 ///< Normally we use the rPOMCP random engine;
            SampleBelief sampleBelief_;           ///< This is a particle belief which is easy to sample
            size_t beliefSize_;                   ///< This is the total number of particles for this belief (sum of each count of the sample belief)
    };

    template <bool UseEntropy>
    BeliefNode<UseEntropy>::BeliefNode() :
            N(0), V(0.0),
            actionsV(0.0), bestAction(0),
            knowledgeMeasure_(0.0) {}

    // Note for ENTROPY implementation:
    // In theory this is wrong as we should update all the entropy terms, one
    // for each different type of particle. In practice we hope this will work
    // anyway, and that there are not going to be huge problems, as each particle
    // should be seen enough times to still keep a decent approximation of its
    // entropy term. Minor errors are ok since this is still an estimation.
    template <>
    void BeliefNode<true>::updateBeliefAndKnowledge(const size_t s) {
        // Remove entropy term for this state from summatory
        knowledgeMeasure_ -= trackBelief_[s].negativeEntropy;
        // Updating belief
        trackBelief_[s].N += 1;
        // Computing new entropy term for this state
        double p = static_cast<double>(trackBelief_[s].N) / static_cast<double>(N+1);
        double newEntropy = p * std::log(p);
        // Update values
        trackBelief_[s].negativeEntropy = newEntropy;
        knowledgeMeasure_ += newEntropy;
    }

    // This is the Max-Belief implementation
    template <>
    void BeliefNode<false>::updateBeliefAndKnowledge(const size_t s) {
        trackBelief_[s].N += 1;

        if ( trackBelief_[s].N > trackBelief_[maxS_].N )
            maxS_ = s;

        knowledgeMeasure_ = static_cast<double>(trackBelief_[maxS_].N) / static_cast<double>(N+1);
    }

    template <bool UseEntropy>
    double BeliefNode<UseEntropy>::getKnowledgeMeasure() const {
        return knowledgeMeasure_;
    }

    template <bool UseEntropy>
    HeadBeliefNode<UseEntropy>::HeadBeliefNode(const size_t A, RandomEngine & rand) : BeliefNode<UseEntropy>(), rand_(&rand) {
        this->children.resize(A);
    }

    template <bool UseEntropy>
    HeadBeliefNode<UseEntropy>::HeadBeliefNode(const size_t A, const size_t beliefSize, const AIToolbox::POMDP::Belief & b, RandomEngine & rand) :
            BeliefNode<UseEntropy>(), rand_(&rand), beliefSize_(beliefSize)
    {
        this->children.resize(A);
        std::unordered_map<size_t, unsigned> generatedSamples;

        size_t S = b.size();
        for ( size_t i = 0; i < beliefSize_; ++i )
            generatedSamples[AIToolbox::sampleProbability(S, b, *rand_)] += 1;

        sampleBelief_.reserve(beliefSize_);
        for ( auto & pair : generatedSamples ) {
            sampleBelief_.emplace_back(pair);
            // Compute entropy here since we don't have a parent in this case (is it really needed?)
            // double p = static_cast<double>(pair.second) / static_cast<double>(beliefSize_);
            // negativeEntropy += p * std::log(p);
        }
    }

    template <bool UseEntropy>
    HeadBeliefNode<UseEntropy>::HeadBeliefNode(const size_t A, BeliefNode<UseEntropy> && bn, RandomEngine & rand) :
            BeliefNode<UseEntropy>(std::move(bn)), rand_(&rand), beliefSize_(0)
    {
        this->children.resize(A);
        sampleBelief_.reserve(this->trackBelief_.size());
        for ( auto & pair : this->trackBelief_ ) {
            sampleBelief_.emplace_back(pair.first, pair.second.N);
            beliefSize_ += pair.second.N;
        }
        TrackBelief<UseEntropy>().swap(this->trackBelief_); // Clear belief memory
    }

    template <bool UseEntropy>
    bool HeadBeliefNode<UseEntropy>::isSampleBeliefEmpty() const {
        return sampleBelief_.empty();
    }

    template <bool UseEntropy>
    size_t HeadBeliefNode<UseEntropy>::sampleBelief() const {
        std::uniform_int_distribution<unsigned> generator(1, beliefSize_);
        int pick = generator(*rand_);

        size_t index = 0;
        while (true) {
            pick -= sampleBelief_[index].second;
            if ( pick < 1 ) return sampleBelief_[index].first;
            ++index;
        }
    }

    template <bool UseEntropy>
    size_t HeadBeliefNode<UseEntropy>::getMostCommonParticle() const {
        // We return the most common particle in the head belief
        size_t bestGuess; unsigned bestGuessCount = 0;
        for ( auto & pair : sampleBelief_ ) {
            if ( pair.second > bestGuessCount ) {
                bestGuessCount = pair.second;
                bestGuess = pair.first;
            }
        }
        return bestGuess;
    }
}

#endif
