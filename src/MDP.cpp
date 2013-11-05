#include <MDPToolbox/MDP.hpp>

#include <cmath>
#include <algorithm>
#include <MDPToolbox/Policy.hpp>

#include <iostream>
using namespace std;

namespace MDPToolbox {
    MDP::MDP(size_t sNum, size_t aNum) : S(sNum), A(aNum) {
        mdp_.resize(S);

        for ( size_t s = 0; s < S; s++ ) {
            mdp_[s].resize(S);

            for ( size_t s1 = 0; s1 < S; s1++ ) {
                mdp_[s][s1].reserve(A);

                for ( size_t a = 0; a < A; a++ ) {
                    mdp_[s][s1].emplace_back( 1.0/S, 0 ); 
                }
            }        
        }
        computePR();
    }

    Policy MDP::valueIteration(double discount, double epsilon, unsigned maxIter, ValueType v1, bool * doneOut) const {
        if ( discount <= 0 || discount > 1 )    throw std::runtime_error("Discount parameter must be in (0,1]");
        if ( epsilon <= 0 )                     throw std::runtime_error("Epsilon must be > 0");
        if ( v1.size() != 0 && v1.size() != S ) throw std::runtime_error("The starting value function has the wrong size");
        
        if ( v1.size() == 0 ) v1.resize(S, 0.0);

        {   // maxIter setup
            unsigned computedMaxIter = ( discount != 1 ) ? valueIterationBoundIter(discount, epsilon, v1) : 1000;
            if ( !maxIter ) 
                maxIter = maxIter ? computedMaxIter : std::min( computedMaxIter, maxIter );
        }
        {   // threshold setup
            epsilon = ( discount != 1 ) ? ( epsilon * ( 1 - discount ) / discount ) : epsilon; 
        }
        
        unsigned iter = 0;
        bool done = false, completed = false;
        ValueType v0;
        Policy p(0,0);

        while ( !done ) {
            iter++;
            v0 = v1;            

            std::tie( v1, p ) = bellmanOperator( discount, v1 );

            std::transform(begin(v1), end(v1), begin(v0), begin(v0), std::minus<double>() );

            double variation; 
            {
                auto minmax = std::minmax_element(begin(v0), end(v0));
                variation = *(minmax.second) - *(minmax.first);
            }
            if ( variation < epsilon ) {
                completed = true;
                done = true;
            }
            else if ( iter > maxIter ) {
                done = true;
            }
        }

        if ( doneOut ) *doneOut = completed;
        
        return p;
    }

    
    void MDP::computePR() {
        pr_.resize(S);
        // for a=1:A; PR(:,a) = sum(P(:,:,a).*R(:,:,a),2); end;
        for ( size_t s = 0; s < S; s++ ) {
            pr_[s].resize(A, 0.0);
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    pr_[s][a] += std::get<Probability>(mdp_[s][s1][a]) * std::get<Reward>(mdp_[s][s1][a]);
                }
            }
        }
    }

    std::tuple<MDP::ValueType, Policy> MDP::bellmanOperator(double discount, const ValueType & v0) const {
        /*
         *  for a=1:A
         *      Q(:,a) = PR(:,a) + discount*P(:,:,a)*Vprev;
         *  end
         *  [V, policy] = max(Q,[],2);
         */
        QType q = pr_; 
        
        for ( size_t s = 0; s < S; s++ )
            for ( size_t s1 = 0; s1 < S; s1++ )
                for ( size_t a = 0; a < A; a++ )
                    q[s][a] += std::get<Probability>(mdp_[s][s1][a]) * discount * v0[s];
        
        ValueType v1(S);
        Policy p(S,A);

        for ( size_t s = 0; s < S; s++ ) {
            auto it = std::max_element(begin(q[s]), end(q[s]));
            p.setPolicy(s, std::distance(begin(q[s]), it));
            v1[s] = *it;
        }

        return std::make_tuple(v1, p);
    }

    unsigned MDP::valueIterationBoundIter(double discount, double epsilon, const ValueType & v0) const {
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
                    h[s1] = std::min(h[s1], std::get<Probability>(mdp_[s][s1][a]));

        double k = 1 - std::accumulate(begin(h), end(h), 0.0);

        ValueType v1;

        std::tie(v1, std::ignore) = bellmanOperator(discount, v0);

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
}
