#include <AIToolbox/MDP/ValueIteration.hpp>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/Solution.hpp>
#include <AIToolbox/Policy.hpp>
#include <AIToolbox/MDP/Utils.hpp>

#include <iostream>

namespace AIToolbox {
    namespace MDP {
        ValueIteration::ValueIteration(double discount, double epsilon, unsigned maxIter, ValueFunction v) : discount_(discount), epsilon_(epsilon), maxIter_(maxIter), vParameter_(v),
                                                                                                             model_(nullptr), S(0), A(0)
        {
            if ( discount_ <= 0 || discount_ > 1 )  throw std::runtime_error("Discount parameter must be in (0,1]");
            if ( epsilon_ <= 0 )                    throw std::runtime_error("Epsilon must be > 0");
        }


        bool ValueIteration::solve(const Model & model, Solution & solution) {
            // Extract necessary knowledge from model so we don't have to pass it around
            S = model.getS();
            A = model.getA();
            model_ = &model;

            // Verify that parameter value function is compatible.
            if ( vParameter_.size() != S ) {
                if ( vParameter_.size() != 0 )
                    std::cerr << "AIToolbox: Size of starting value function in ValueIteration::solve() is incorrect, ignoring...\n";
                v1_ = ValueFunction(S, 0.0);
            }
            else
                v1_ = vParameter_;

            auto pr = computePR();
            {   // maxIter setup
                unsigned computedMaxIter = valueIterationBoundIter(pr);
                if ( !maxIter_ ) {
                    maxIter_ = discount_ != 1.0 ? computedMaxIter : 1000;
                }
                else {
                    maxIter_ = ( discount_ != 1.0 && maxIter_ > computedMaxIter ) ? computedMaxIter : maxIter_;
                }
            }
            {   // threshold setup
                epsilon_ = ( discount_ != 1 ) ? ( epsilon_ * ( 1 - discount_ ) / discount_ ) : epsilon_;
            }

            //cout << "I'm starting now..\n";

            unsigned iter = 0;
            bool done = false, completed = false;

            ValueFunction v0 = v1_;

            while ( !done ) {
                iter++;
                //std::cout << "Iteration: " << iter << "\n";
                v0 = v1_;

                bellmanOperator( pr, v1_ );

                std::transform(std::begin(v1_), std::end(v1_), std::begin(v0), std::begin(v0), std::minus<double>() );

                double variation;
                {
                    auto minmax = std::minmax_element(std::begin(v0), std::end(v0));
                    variation = *(minmax.second) - *(minmax.first);
                }
                //std::cout << "    Variation: " << variation << "\n";
                if ( variation < epsilon_ ) {
                    completed = true;
                    done = true;
                }
                else if ( iter > maxIter_ ) {
                    done = true;
                }
            }

            solution.setValueFunction(v1_);
            solution.setQFunction(makeQFunction(pr));
            solution.setPolicy(makePolicy(S, A, solution.getQFunction()));

            model_ = nullptr;
            S = A = 0;

            return completed;
        }

        ValueIteration::PRType ValueIteration::computePR() const {
            // for a=1:A; PR(:,a) = sum(P(:,:,a).*R(:,:,a),2); end;
            PRType pr(boost::extents[S][A]);

            for ( size_t s = 0; s < S; s++ ) {
                for ( size_t s1 = 0; s1 < S; s1++ ) {
                    for ( size_t a = 0; a < A; a++ ) {
                        pr[s][a] += model_->getTransitionFunction()[s][s1][a] * model_->getRewardFunction()[s][s1][a];
                    }
                }
            }
            return pr;
        }

        QFunction ValueIteration::makeQFunction(const PRType & pr) const {
            QFunction q = pr;

            for ( size_t s = 0; s < S; s++ )
                for ( size_t s1 = 0; s1 < S; s1++ )
                    for ( size_t a = 0; a < A; a++ )
                        q[s][a] += model_->getTransitionFunction()[s][s1][a] * discount_ * v1_[s1];
            return q;
        }

        void ValueIteration::bellmanOperator(const PRType & pr, ValueFunction & vOut) const {
            /*
             *  for a=1:A
             *      Q(:,a) = PR(:,a) + discount*P(:,:,a)*Vprev;
             *  end
             *  [V, policy] = max(Q,[],2);
             */
            QFunction q = makeQFunction(pr);

            for ( size_t s = 0; s < S; s++ ) {
                auto it = std::max_element(std::begin(q[s]), std::end(q[s]));
                vOut[s] = *it;
            }
        }

        unsigned ValueIteration::valueIterationBoundIter(const PRType & pr) const {
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
                        h[s1] = std::min(h[s1], model_->getTransitionFunction()[s][s1][a]);

            double k = 1 - std::accumulate(std::begin(h), std::end(h), 0.0);

            ValueFunction v;

            bellmanOperator(pr, v);

            std::transform(std::begin(v), std::end(v), std::begin(v1_), std::begin(v), std::minus<double>() );

            double variation;
            {
                auto minmax = std::minmax_element(std::begin(v), std::end(v));
                variation = *(minmax.second) - *(minmax.first);
            }

            return std::ceil (
                    std::log( (epsilon_*(1-discount_)/discount_) / variation ) / std::log(discount_*k));
        }
    }
}
