#ifndef AI_TOOLBOX_IMPL_CASSANDRA_PARSER_HEADER_FILE
#define AI_TOOLBOX_IMPL_CASSANDRA_PARSER_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

namespace AIToolbox::Impl {
    /**
     * @brief This class can parse files containing MDPs and POMDPs in the Cassandra file format.
     */
    class CassandraParser {
        public:
            using MDPVals = std::tuple<size_t, size_t, DumbMatrix3D, DumbMatrix3D, double>;
            using POMDPVals = std::tuple<size_t, size_t, size_t, DumbMatrix3D, DumbMatrix3D, DumbMatrix3D, double>;

            /**
             * @brief Basic constructor.
             */
            CassandraParser();

            /**
             * @brief This function parses the input following Cassandra's rules.
             *
             * This function only parses for information regarding an MDP, so
             * number of states, number of actions, discount and
             * transition/reward functions.
             *
             * Any problems during parsing result in an std::runtime_error.
             *
             * The output is returned as a set of values which if needed can be
             * used to build an MDP::Model.
             *
             * No checks are done here regarding the consistency of the read
             * data (transition probabilities, etc).
             *
             * @param input The input stream to parse.
             *
             * @return A tuple containing S, A, T, R, and discount of the parsed MDP.
             */
            MDPVals parseMDP(std::istream & input);

            /**
             * @brief This function parses the input following Cassandra's rules.
             *
             * this function only parses for information regarding an pomdp, so
             * number of states, number of actions, number of observations,
             * discount and transition/reward/observation functions.
             *
             * Any problems during parsing result in an std::runtime_error.
             *
             * The output is returned as a set of values which if needed can be
             * used to build an POMDP::Model.
             *
             * No checks are done here regarding the consistency of the read
             * data (transition probabilities, etc).
             *
             * @param input The input stream to parse.
             *
             * @return A tuple containing S, A, O, T, R, W, and discount of the parsed POMDP.
             */
            POMDPVals parsePOMDP(std::istream & input);

        private:
            using IDMap = std::unordered_map<std::string, size_t>;
            using ActionMap = std::unordered_map<std::string, std::function<void(const std::string &)>>;
            using Tokens = std::vector<std::string>;

            /**
             * @brief This function parses the preamble from the input.
             *
             * This function is called for both MDPs and POMDPs, and parses all it can, without caring what it finds and what it does not.
             *
             * @param input The input stream to parse.
             */
            void parseModelInfo(std::istream & input);

            /**
             * @brief This function extracts ids from numbers or string tokens.
             *
             * @param line The line that contains the tokens.
             * @param map Where to store tokens if they are not specified in numerical form.
             *
             * @return The number of tokens parsed.
             */
            size_t extractIDs(const std::string & line, IDMap & map);

            /**
             * @brief This function splits the input string into tokens, divided by the input character list.
             *
             * @param str The string to split.
             * @param list The list of characters to split tokens.
             *
             * @return The list of obtained tokens.
             */
            Tokens tokenize(const std::string & str, const char * list);

            /**
             * @brief This function returns which indeces to set when parsing matrix declarations.
             *
             * It can handle both '*' specifications, and string specifications
             * which use named tokens declared in the preamble.
             *
             * @param str The string that contains the indeces.
             * @param map The map that contains the string representations of the tokens.
             * @param max The max number that is allowed to be parsed.
             *
             * @return The list of indeces that apply.
             */
            std::vector<size_t> parseIndeces(const std::string & str, const IDMap & map, size_t max);

            /**
             * @brief This function parses a vector of length N from the inputs.
             *
             * This function does a simple conversion of token to double.
             *
             * @param begin The beginning of the range.
             * @param end The end of the range.
             * @param N The number of tokens to parse.
             *
             * @return A vector with the parsed numbers.
             */
            std::vector<double> parseVector(Tokens::const_iterator begin, Tokens::const_iterator end, size_t N);

            /**
             * @brief This function parses a vector of length N from the input string.
             *
             * This function assumes that the elements of the vector are
             * separated by space.
             *
             * @param str The string to be parsed.
             * @param N The number of tokens to extract.
             *
             * @return The parsed vector.
             */
            std::vector<double> parseVector(const std::string & str, size_t N);

            /**
             * @brief This function parses an entry for a specific matrix.
             *
             * This function is used for both transition and observation
             * functions, since the syntax for both is the same.
             *
             * Since both are indexed by action in the same way, we don't need
             * input for that.
             *
             * @param M The matrix to be read in.
             * @param d1map The list of id string tokens for the first dimension of the matrix.
             * @param d3map The list of id string tokens for the first dimension of the matrix.
             */
            void processMatrix(DumbMatrix3D & M, const IDMap & d1map, const IDMap & d3map);

            /**
             * @brief This function processes a reward function entry.
             *
             * We only support entries that do not specify observations, and
             * thus, per syntax, must specify values one by one.
             *
             * @param R The reward matrix to be read in.
             */
            void processReward(DumbMatrix3D & R);

            // Storage for lines which are not empty and not used in the preamble.
            std::vector<std::string> lines_;
            size_t i_;

            // Storage for input preamble.
            // We need these so that we can store lambdas in the initMap_ with
            // the same signature.
            size_t S_, A_, O_;
            double discount_;

            // These are actions to perform for the preamble.
            ActionMap initMap_;

            // These contain the stringToken->id maps.
            IDMap stateMap_;
            IDMap actionMap_;
            IDMap observationMap_;
    };
}

#endif
