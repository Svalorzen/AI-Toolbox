#ifndef AI_TOOLBOX_POMDP_GAPMIN_HEADER_FILE
#define AI_TOOLBOX_POMDP_GAPMIN_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>

#include <AIToolbox/POMDP/Algorithms/BlindStrategies.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the GapMin algorithm.
     */
    class GapMin {
        public:
            /**
             * @brief Basic constructor.
             */
            GapMin();

            /**
             * @brief This function solves a POMDP::Model approximately.
             */
            template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
            std::tuple<double, ValueFunction> operator()(const M & model);

        private:
            double epsilon_;
    };

    template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
    std::tuple<double, ValueFunction> GapMin::operator()(const M & m) {
        BlindStrategies bs(1000000, epsilon_);
        auto lbVList = bs(m, true);
        {
            const auto unwrap = +[](VEntry & ve) -> MDP::Values & {return std::get<VALUES>(ve);};
            const auto rbegin = boost::make_transform_iterator(std::begin(lbVList), unwrap);
            const auto rend   = boost::make_transform_iterator(std::end  (lbVList), unwrap);

            lbVList.erase(extractDominated(m.getS(), rbegin, rend).base(), std::end(lbVList));
        }
    }
}

#endif

