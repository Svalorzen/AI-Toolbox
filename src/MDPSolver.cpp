
namespace AIToolbox {
    bool MDPSolver::valueIteration(double discount, double epsilon, unsigned maxIter, ValueFunction v1 ) {
        if ( discount <= 0 || discount > 1 )    throw std::runtime_error("Discount parameter must be in (0,1]");
        if ( epsilon <= 0 )                     throw std::runtime_error("Epsilon must be > 0");
        if ( v1.size() != 0 && v1.size() != S ) throw std::runtime_error("The starting value function has the wrong size");

        if ( v1.size() == 0 ) v1.resize(S, 0.0);

        computePR();

        {   // maxIter setup
            unsigned computedMaxIter = valueIterationBoundIter(discount, epsilon, v1);
            if ( !maxIter ) {
                maxIter = discount != 1.0 ? computedMaxIter : 1000;
            }
            else {
                maxIter = ( discount != 1.0 && maxIter > computedMaxIter ) ? computedMaxIter : maxIter;
            }
        }
        {   // threshold setup
            epsilon = ( discount != 1 ) ? ( epsilon * ( 1 - discount ) / discount ) : epsilon;
        }

        //cout << "I'm starting now..\n";

        unsigned iter = 0;
        bool done = false, completed = false;
        ValueFunction v0;

        while ( !done ) {
            iter++;
            //std::cout << "Iteration: " << iter << "\n";
            v0 = v1;

            std::tie( q_, v1, policy_ ) = bellmanOperator( discount, v1 );

            std::transform(std::begin(v1), std::end(v1), std::begin(v0), std::begin(v0), std::minus<double>() );

            double variation;
            {
                auto minmax = std::minmax_element(std::begin(v0), std::end(v0));
                variation = *(minmax.second) - *(minmax.first);
            }
            //std::cout << "    Variation: " << variation << "\n";
            if ( variation < epsilon ) {
                completed = true;
                done = true;
            }
            else if ( iter > maxIter ) {
                done = true;
            }
        }

        v_ = v1;

        return completed;
    }

    void MDPSolver::dynaQ(size_t s, size_t a, double discount) {
        size_t s1;
        double rew;

        std::tie(s1, rew) = model_.sample(s,a); 

        updateQ(s, s1, a, rew, discount);
    }
/*
    void MDPSolver::prioritizedSweeping(double discount, double threshold) {
        size_t s, a;
        // TODO: Get s and a from queue

        dynaQ(s, a, discount);
        updatePrioritizedSweepingQueue(s, discount, threshold);
    }
*/
    void MDPSolver::updateQ(size_t s, size_t s1, size_t a, double rew, double discount) {
        q_[s][a] += discount * ( rew * (*std::max_element(std::begin(q_[s1]),std::end(q_[s1]))) - q_[s][a] );
    }
/*
    void MDPSolver::updatePrioritizedSweepingQueue( size_t state, double discount, double threshold ) {
        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t a = 0; a < A; a++ ) {
                if ( transitions_[s][state][a] != 0.0 ) {
                    double p = std::fabs(rewards_[s][state][a] + discount * (*std::max_element(std::begin(q_[state]), std::end(q_[state]))) - q_[s][a]);
                    if ( p > threshold )
                        return;
                }
            }
        }
    }
*/
    void MDPSolver::computePR() {
        // for a=1:A; PR(:,a) = sum(P(:,:,a).*R(:,:,a),2); end;
        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    pr_[s][a] += model_.getTransitionFunction()[s][s1][a] * model_.getRewardFunction()[s][s1][a];
                }
            }
        }
    }

    std::tuple<MDPSolver::QFunction, MDPSolver::ValueFunction, Policy> MDPSolver::bellmanOperator(double discount, const ValueFunction & v0) const {
        /*
         *  for a=1:A
         *      Q(:,a) = PR(:,a) + discount*P(:,:,a)*Vprev;
         *  end
         *  [V, policy] = max(Q,[],2);
         */
        QFunction q = pr_;

        for ( size_t s = 0; s < S; s++ )
            for ( size_t s1 = 0; s1 < S; s1++ )
                for ( size_t a = 0; a < A; a++ )
                    q[s][a] += model_.getTransitionFunction()[s][s1][a] * discount * v0[s1];

        ValueFunction v1(S);
        Policy p(S,A);

        for ( size_t s = 0; s < S; s++ ) {
            auto it = std::max_element(std::begin(q[s]), std::end(q[s]));
            p.setPolicy(s, static_cast<size_t>(std::distance(std::begin(q[s]), it)));
            v1[s] = *it;
        }

        return std::make_tuple(q, v1, p);
    }

    unsigned MDPSolver::valueIterationBoundIter(double discount, double epsilon, const ValueFunction & v0) const {
        /*
         *  for ss=1:S; h(ss) = min(min(P(:,ss,:))); end;
         *  k = 1 - sum(h);
         *  V1 = mdp_bellman_operator(P,PR,discount,V0);
         *  max_iter = log ( (epsilon*(1-discount)/discount) / mdp_span(V1-V0) ) / log(discount*k);
         *
         *  max_iter = ceil(max_iter);
         */
        std::vector<double> h(S, 0.0);

        for ( size_t s = 0; s < S; s++ )
            for ( size_t s1 = 0; s1 < S; s1++ )
                for ( size_t a = 0; a < A; a++ )
                    h[s1] = std::min(h[s1], model_.getTransitionFunction()[s][s1][a]);

        double k = 1 - std::accumulate(std::begin(h), std::end(h), 0.0);

        ValueFunction v1;

        std::tie(std::ignore, v1, std::ignore) = bellmanOperator(discount, v0);

        std::transform(std::begin(v1), std::end(v1), std::begin(v0), std::begin(v1), std::minus<double>() );

        double variation;
        {
            auto minmax = std::minmax_element(std::begin(v1), std::end(v1));
            variation = *(minmax.second) - *(minmax.first);
        }

        return std::ceil (
                std::log( (epsilon*(1-discount)/discount) / variation ) / std::log(discount*k));
    }


    // EPSILON GREEDY
        if ( epsilon < 1.0 ) {
            double greedy = sampleDistribution_(rand_);
            if ( greedy > epsilon ) {
                // RANDOM!
                return randomDistribution_(rand_);
            }
        }

        Policy makeGreedyPolicy(size_t S, size_t A, const QFunction & q) {
            Policy p(S,A);
            std::vector<double> probs(S);
            for ( size_t s = 0; s < S; s++ ) {
                double max = *std::max_element(std::begin(q[s]), std::end(q[s]));
                for ( size_t a = 0; a < A; a++ ) {
                    probs[a] = static_cast<double>(q[s][a] == max);
                }
                p.setStatePolicy(s, probs);
            }
            return p;
        }

        void updatePolicy(Policy & p, size_t s, const QFunction & q) {
            double max = *std::max_element(std::begin(q[s]), std::end(q[s]));
            std::vector<double> probs(p.getS());
            for ( size_t a = 0; a < p.getA(); a++ ) {
                probs[a] = static_cast<double>(q[s][a] == max);
            }
            p.setStatePolicy(s, probs);
        }
}
