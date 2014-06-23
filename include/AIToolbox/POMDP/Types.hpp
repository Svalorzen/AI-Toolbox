#ifndef AI_TOOLBOX_POMDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_POMDP_TYPES_HEADER_FILE

#include <utility>
#include <vector>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {
        using Belief            = std::vector<double>;

        /**
         * @defgroup POMDPVF POMDP Value Functions.
         *
         * POMDP ValueFunctions are complicated. In fact, they are trees. At each
         * belief, you have a particular value function, which also depends on any
         * future observation you might encounter, and must change "value" accordingly.
         * For example, a high valued belief might in the end turn out bad due to
         * repeated "bad" observations. At the same time, for each particular block
         * of values, we want to save the single action that will result in the
         * "actuation" of that particular value.
         *
         * We avoid storing a POMDP ValueFunction as a true tree, mostly due to the
         * fact that most operations like search and update are done on a timestep
         * basis, as in, specific tree depth. Thus the layout is arranged as follows:
         *
         * A VEntry contains the MDP::Values for its specific Belief range. At any
         * belief it can be used to compute, via dot product, the true value of that
         * belief. Together with it, it contains an action index, for the action that
         * results in the actuation of those particular values, and a vector of
         * indeces into the previous VList. There are going to be |O| entries in this
         * vector. Some observations are however impossible from certain beliefs. In
         * theory, those should never be accessed, and those entries will just keep
         * the value of zero to keep things simple.
         *
         * A VList is a slice of the final tree with respect to depth, as in all
         * ValueFunctions for a certain timestep t.
         *
         * A ValueFunction is the final tree keeping all VLists together.
         *
         * QFunctions may be defined later, however since POMDP ValueFunctions are already
         * pretty costly in terms of space, in general there's little sense in storing them.
         *
         * @{
         */

        using VObs          = std::vector<size_t>;
        using VEntry        = std::tuple<MDP::Values, size_t, VObs>;
        enum {
            VALUES = 0,
            ACTION = 1,
            OBS    = 2,
        };
        using VList         = std::vector<VEntry>;
        using ValueFunction = std::vector<VList>;

        /** @}  */

        /**
         * @brief This struct represents the required interface for a generative MDP.
         *
         * This struct is used to check interfaces of classes in templates.  In
         * particular, this struct tests for the interface of a generative MDP
         * model.  The interface must be implemented and be public in the
         * parameter class. The interface is the following:
         *
         * - std::tuple<size_t, size_t, double> sampleSOR(size_t s, size_t a) const : Returns a sampled state-observation-reward tuple from (s,a)
         *
         * is_generative_model<M>::value will be equal to true is M implements the interface,
         * and false otherwise.
         *
         * Note that, at least for now, we can avoid asking this generative
         * model for the total number of observation possible, because they are
         * not required as parameters for the functions, but just returned.
         * This may change in future though, depending on algorithms'
         * requirements.
         *
         * @tparam M The class to test for the interface.
         */
        template <typename M>
        struct is_generative_model {
            private:
                template <typename Z> static auto test(int) -> decltype(

                        static_cast<std::tuple<size_t,size_t,double> (Z::*)(size_t,size_t) const>      (&Z::sampleSOR),

                        std::true_type()
                );

                template <typename Z> static auto test(...) -> std::false_type;

            public:
                enum { value = std::is_same<decltype(test<M>(0)),std::true_type>::value };
        };

        /**
         * @brief This struct represents the required interface for a POMDP Model.
         *
         * This struct is used to check interfaces of classes in templates.
         * In particular, this struct tests for the interface of a POMDP model.
         * The interface must be implemented and be public in the parameter
         * class. The interface is the following:
         *
         * - size_t getO() const : Returns the number of observations of the Model.
         * - double getObservationProbability(size_t s1, size_t a, size_t o) : Returns the probability for observation o after action a and final state s1.
         *
         * In addition the POMDP needs to respect the interface for the POMDP generative
         * model and the MDP Model.
         *
         * \sa is_generative_model
         * \sa MDP::is_model
         *
         * is_model<M>::value will be equal to true is M implements the interface,
         * and false otherwise.
         *
         * @tparam T The class to test for the interface.
         */
        template <typename T>
        struct is_model {
            private:
                template <typename Z> static auto test(int) -> decltype(

                        static_cast<size_t (Z::*)() const>                      (&Z::getO),
                        static_cast<double (Z::*)(size_t,size_t,size_t) const>  (&Z::getObservationProbability),

                        std::true_type()
                );

                template <typename> static auto test(...) -> std::false_type;

            public:
                enum { value = std::is_same<decltype(test<T>(0)),std::true_type>::value && is_generative_model<T>::value && MDP::is_model<T>::value };
        };
    }
}

#endif
