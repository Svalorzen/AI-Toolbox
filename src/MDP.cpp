#include <MDPToolbox/MDP.hpp>

#include <cmath>
#include <algorithm>
#include <chrono>

#include <iostream>
using namespace std;

namespace MDPToolbox {
    MDP::MDP(size_t sNum, size_t aNum) : S(sNum), A(aNum), transitions_(boost::extents[S][S][A]), rewards_(boost::extents[S][S][A]), pr_(boost::extents[S][A]),
                                         q_(boost::extents[S][A]), v_(S,0.0), policy_(S,A),
                                         rand_(std::chrono::system_clock::now().time_since_epoch().count()), sampleDistribution_(0.0, 1.0)
    {
        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    transitions_[s][s1][a] = 1.0/S;
                    rewards_[s][s1][a] = 0.0;
                }
            }
        }
        computePR();
    }

    bool MDP::valueIteration(double discount, double epsilon, unsigned maxIter, ValueFunction v1 ) {
        if ( discount <= 0 || discount > 1 )    throw std::runtime_error("Discount parameter must be in (0,1]");
        if ( epsilon <= 0 )                     throw std::runtime_error("Epsilon must be > 0");
        if ( v1.size() != 0 && v1.size() != S ) throw std::runtime_error("The starting value function has the wrong size");

        if ( v1.size() == 0 ) v1.resize(S, 0.0);

        {   // maxIter setup
            unsigned computedMaxIter = ( discount != 1 ) ? valueIterationBoundIter(discount, epsilon, v1) : 1000;
            if ( !maxIter )
                maxIter = ! maxIter ? computedMaxIter : std::min( computedMaxIter, maxIter );
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
            //std::cout << "Iteration: " << iter;
            v0 = v1;

            std::tie( q_, v1, policy_ ) = bellmanOperator( discount, v1 );

            std::transform(begin(v1), end(v1), begin(v0), begin(v0), std::minus<double>() );

            double variation;
            {
                auto minmax = std::minmax_element(begin(v0), end(v0));
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
        std::copy(begin(v1), end(v1), begin(v_));

        return completed;
    }


    void MDP::computePR() {
        // for a=1:A; PR(:,a) = sum(P(:,:,a).*R(:,:,a),2); end;
        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    pr_[s][a] += transitions_[s][s1][a] * rewards_[s][s1][a];
                }
            }
        }
    }

    std::tuple<MDP::QFunction, MDP::ValueFunction, Policy> MDP::bellmanOperator(double discount, const ValueFunction & v0) const {
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
                    q[s][a] += transitions_[s][s1][a] * discount * v0[s1];

        ValueFunction v1(S);
        Policy p(S,A);

        for ( size_t s = 0; s < S; s++ ) {
            auto it = std::max_element(begin(q[s]), end(q[s]));
            p.setPolicy(s, static_cast<size_t>(std::distance(begin(q[s]), it)));
            v1[s] = *it;
        }

        return std::make_tuple(q, v1, p);
    }

    unsigned MDP::valueIterationBoundIter(double discount, double epsilon, const ValueFunction & v0) const {
        /*
         *  for ss=1:S; h(ss) = min(min(P(:,ss,:))); end;
         *  k = 1 - sum(h);
         *  V1 = mdp_bellman_operator(P,PR,discount,V0);
         *  max_iter = log ( (epsilon*(1-discount)/discount) / mdp_span(V1-V0) ) / log(discount*k);
         *
         *  max_iter = ceil(max_iter);
         */
        std::vector<double> h(S);
        for ( size_t s = 0; s < S; s++ )
            for ( size_t s1 = 0; s1 < S; s1++ )
                for ( size_t a = 0; a < A; a++ )
                    h[s1] = std::min(h[s1], transitions_[s][s1][a]);

        double k = 1 - std::accumulate(begin(h), end(h), 0.0);

        ValueFunction v1;

        std::tie(std::ignore, v1, std::ignore) = bellmanOperator(discount, v0);

        std::transform(begin(v1), end(v1), begin(v0), begin(v1), std::minus<double>() );

        double variation;
        {
            auto minmax = std::minmax_element(begin(v1), end(v1));
            variation = *(minmax.second) - *(minmax.first);
        }

        return std::ceil (
                std::log( (epsilon*(1-discount)/discount) / variation ) / log(discount*k));
    }

    size_t MDP::getS() const {
        return S;
    }

    size_t MDP::getA() const {
        return A;
    }

    std::tuple<size_t, double> MDP::sample(size_t s, size_t a) const {
        double p = sampleDistribution_(rand_);

        for ( size_t s1 = 0; s1 < S; s1++ ) {
            if ( transitions_[s][s1][a] > p ) return std::make_tuple(s1, rewards_[s][s1][a]);
            p -= transitions_[s][s1][a];
        }
        return std::make_tuple(S-1, rewards_[s][S-1][a]);
    }

    const Policy & MDP::getPolicy() const {
        return policy_;
    }

    const MDP::ValueFunction & MDP::getValueFunction() const {
        return v_;
    }

    const MDP::QFunction & MDP::getQFunction() const {
        return q_;
    }
}
