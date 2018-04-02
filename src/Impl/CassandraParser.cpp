#include <AIToolbox/Impl/CassandraParser.hpp>

#include <numeric>
#include <istream>
#include <algorithm>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

namespace AIToolbox::Impl {
    CassandraParser::CassandraParser() {
        // Assign an action to parse each value for the preambles. Lines parsed
        // in the preamble are parsed before the others.
        initMap_["values"] = [](const std::string &){};
        initMap_["states"] = [this](const std::string & line){
            S = extractIDs(line, stateMap_);
        };
        initMap_["actions"] = [this](const std::string & line){
            A = extractIDs(line, actionMap_);
        };
        initMap_["observations"] = [this](const std::string & line){
            O = extractIDs(line, observationMap_);
        };
        initMap_["discount"] = [this](const std::string & line){
            discount = std::stod(tokenize(line, ":").at(1));
        };
    }

    CassandraParser::MDPVals CassandraParser::parseMDP(std::istream & input) {
        // Parse preamble.
        parseModelInfo(input);

        if (!S || !A)
            throw std::runtime_error("MDP definition is incomplete");

        // Init matrices to store data.
        initMatrices();

        for (i_ = 0; i_< lines_.size(); ++i_) {
            auto & line = lines_[i_];

            if (boost::starts_with(line, "T")) {
                processMatrix(T, stateMap_, stateMap_);
                continue;
            }

            if (boost::starts_with(line, "R")) {
                processReward();
                continue;
            }
        }

        return MDPVals(S, A, T, R, discount);
    }

    CassandraParser::POMDPVals CassandraParser::parsePOMDP(std::istream & input) {
        // Parse preamble.
        parseModelInfo(input);

        if (!S || !A || !O)
            throw std::runtime_error("POMDP definition is incomplete");

        // Init matrices to store data.
        initMatrices();

        for (i_ = 0; i_< lines_.size(); ++i_) {
            auto & line = lines_[i_];

            if (boost::starts_with(line, "T")) {
                processMatrix(T, stateMap_, stateMap_);
                continue;
            }

            if (boost::starts_with(line, "O")) {
                processMatrix(W, stateMap_, observationMap_);
                continue;
            }

            if (boost::starts_with(line, "R")) {
                processReward();
                continue;
            }
        }

        return POMDPVals(S, A, O, T, R, W, discount);
    }

    // ############################
    // ####  PRIVATE FUNCTIONS  ###
    // ############################

    void CassandraParser::parseModelInfo(std::istream & input) {
        // This is the same for both MDP and POMDP, and we don't really care
        // here what we find or not.

        // Clear all variables that may have been setup. We don't have to clear
        // the matrices since we do that later during initialization.
        lines_.clear();
        S = 0, A = 0, O = 0;
        discount = 1.0;

        for(std::string line; std::getline(input, line); ) {
            boost::trim(line);
            if (line == "") continue;

            bool parsed = false;
            for (auto it : initMap_) {
                if (boost::starts_with(line, it.first)) {
                    it.second(line);
                    parsed = true;
                    break;
                }
            }

            if (!parsed)
                lines_.push_back(std::move(line));
        }
    }

    void CassandraParser::initMatrix(DumbMatrix3D & M, size_t D1, size_t D2, size_t D3) {
        M.resize(D1);
        for (size_t i = 0; i < D1; ++i) {
            M[i].resize(D2);
            for (size_t j = 0; j < D2; ++j) {
                M[i][j].resize(D3);
                std::fill(std::begin(M[i][j]), std::end(M[i][j]), 0);
            }
        }
    }

    void CassandraParser::initMatrices() {
        if (S && A) {
            initMatrix(T, S, A, S);
            initMatrix(R, S, A, S);
            if (O)
                initMatrix(W, S, A, O);
        }
    }

    size_t CassandraParser::extractIDs(const std::string & line, IDMap & map) {
        map.clear();

        auto split = tokenize(line, ":");
        auto ids = tokenize(split.at(1), " ");

        // Try the number way
        if (ids.size() == 1) {
            try {
                return std::stoul(ids[0]);
            } catch (std::invalid_argument) {}
        }

        for (size_t i = 0; i < ids.size(); ++i)
            map[boost::trim_copy(ids[i])] = i;

        return ids.size();
    }

    std::vector<std::string> CassandraParser::tokenize(const std::string & str, const char * list) {
        using Tokenizer = boost::tokenizer<boost::char_separator<char>>;
        boost::char_separator<char> sep(list);

        Tokenizer parser(str, sep);

        std::vector<std::string> tokens;
        std::copy(std::begin(parser), std::end(parser), std::back_inserter(tokens));
        for (auto & str : tokens)
            boost::trim(str);

        return tokens;
    }

    std::vector<size_t> CassandraParser::parseIndeces(const std::string & str, const IDMap & map, size_t max) {
        std::vector<size_t> retval;

        if (str == "*") {
            retval.resize(max);
            std::iota(std::begin(retval), std::end(retval), 0);
        } else {
            if (auto it = map.find(str); it != std::end(map)) {
                retval.push_back(it->second);
            } else {
                size_t val = std::stoul(str);
                if (val >= max) throw std::runtime_error("Input value too high");
                retval.push_back(val);
            }
        }
        return retval;
    }

    CassandraParser::DumbMatrix1D CassandraParser::parseVector(std::vector<std::string>::iterator begin, std::vector<std::string>::iterator end, size_t N) {
        if (std::distance(begin, end) != (int)N)
            throw std::runtime_error("Wrong number of elements when parsing vector.");

        DumbMatrix1D retval;
        for (; begin < end; ++begin)
            retval.push_back(std::stod(*begin));

        return retval;
    }

    CassandraParser::DumbMatrix1D CassandraParser::parseVector(const std::string & str, size_t N) {
        auto tokens = tokenize(str, " ");
        return parseVector(std::begin(tokens), std::end(tokens), N);
    }

    void CassandraParser::processMatrix(DumbMatrix3D & M, const IDMap & d1map, const IDMap & d3map) {
        const std::string & str = lines_[i_];

        size_t D1 = M.size();
        size_t D3 = M[0][0].size();

        switch (std::count(std::begin(str), std::end(str), ':')) {
            case 3: {
                // M: <action> : <start-state> : <end-state> <prob>
                auto tokens = tokenize(str, ": ");

                // Action is first both in transition and observation
                auto av  = parseIndeces(tokens[1], actionMap_, A);
                auto d1v  = parseIndeces(tokens[2], d1map, D1);
                auto d3v  = parseIndeces(tokens[3], d3map, D3);
                auto val = std::stod(tokens[4]);

                for (auto d1 : d1v)
                    for (auto a : av)
                        for (auto d3 : d3v)
                            M[d1][a][d3] = val;
                break;
            }
            case 2: {
                // M: <action> : <start-state>
                // Here we need to read a vector
                auto tokens = tokenize(str, ": ");

                auto av  = parseIndeces(tokens[1], actionMap_, A);
                auto d1v  = parseIndeces(tokens[2], d1map, D1);

                DumbMatrix1D v;
                if (tokens.size() == 3 + D3) {
                    // Parse at the end
                    v = parseVector(std::begin(tokens) + 3, std::end(tokens), D3);
                } else if (tokens.size() == 3) {
                    // Parse next line
                    v = parseVector(lines_.at(++i_), D3);
                } else {
                    std::runtime_error("Parsing error: wrong number of arguments in '" + str + "'");
                }
                for (auto d1 : d1v)
                    for (auto a : av)
                        M[d1][a] = v;
                break;
            }
            case 1: {
                // M: <action>
                // Here we need to read a whole 2D table
                auto tokens = tokenize(str, ": ");
                auto av  = parseIndeces(tokens[1], actionMap_, A);

                for (size_t d1 = 0; d1 < D1; ++d1) {
                    auto v = parseVector(lines_.at(++i_), D3);

                    for (auto a : av)
                        M[d1][a] = v;
                }
                break;
            }
            default: throw std::runtime_error("Parsing error: wrong number of ':' in '" + str + "'");
        }
    }

    void CassandraParser::processReward() {
        const std::string & str = lines_[i_];

        switch (std::count(std::begin(str), std::end(str), ':')) {
            case 4: {
                // R: <action> : <start-state> : <end-state> : <obs> <prob>
                auto tokens = tokenize(str, ": ");

                // Action is first both in transition and observation
                auto av   = parseIndeces(tokens[1], actionMap_, A);
                auto sv   = parseIndeces(tokens[2], stateMap_,  S);
                auto s1v  = parseIndeces(tokens[3], stateMap_,  S);
                auto val = std::stod(tokens[5]);

                for (auto s : sv)
                    for (auto a : av)
                        for (auto s1 : s1v)
                            R[s][a][s1] = val;
                break;
            }
            default: throw std::runtime_error("Parsing error: wrong number of ':' in '" + str + "'");
        }
    }
}
