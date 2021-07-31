#define BOOST_TEST_MODULE UtilsIO
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/IO.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

namespace ai = AIToolbox;

static ai::RandomEngine rndEng;

ai::Vector makeRandomVector(size_t s) {

    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    return ai::Vector::NullaryExpr(s, [&](){return dist(rndEng);});
}

ai::Matrix2D makeRandomMatrix2D(size_t rows, size_t cols) {
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    return ai::Matrix2D::NullaryExpr(rows, cols, [&](){return dist(rndEng);});
}

ai::SparseMatrix2D makeRandomSparseMatrix2D(size_t rows, size_t cols) {
    std::uniform_int_distribution<unsigned> eNumGen(0, rows * cols);
    const auto num = eNumGen(rndEng);

    std::vector<char> nonZero(rows * cols);
    for (size_t i = 0; i < num; ++i)
        nonZero[i] = 1;

    std::shuffle(std::begin(nonZero), std::end(nonZero), rndEng);

    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(num);

    size_t counter = 0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (nonZero[counter++])
                entries.emplace_back(i, j, dist(rndEng));

    ai::SparseMatrix2D retval(rows, cols);
    retval.setFromTriplets(std::begin(entries), std::end(entries));

    return retval;
}

ai::Table2D makeRandomTable2D(size_t rows, size_t cols) {
    std::uniform_int_distribution<unsigned long> dist(0, 100);

    return ai::Table2D::NullaryExpr(rows, cols, [&](){return dist(rndEng);});
}

ai::SparseTable2D makeRandomSparseTable2D(size_t rows, size_t cols) {
    std::uniform_int_distribution<unsigned> eNumGen(0, rows * cols);
    const auto num = eNumGen(rndEng);

    std::vector<char> nonZero(rows * cols);
    for (size_t i = 0; i < num; ++i)
        nonZero[i] = 1;

    std::shuffle(std::begin(nonZero), std::end(nonZero), rndEng);

    std::uniform_int_distribution<unsigned long> dist(0, 100);

    std::vector<Eigen::Triplet<unsigned long>> entries;
    entries.reserve(num);

    size_t counter = 0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (nonZero[counter++])
                entries.emplace_back(i, j, dist(rndEng));

    ai::SparseTable2D retval(rows, cols);
    retval.setFromTriplets(std::begin(entries), std::end(entries));

    return retval;
}

BOOST_AUTO_TEST_CASE( doubleWrite ) {
    const double d0 = 0.0;
    const double d1 = 1.0;
    const double d2 = 2.0;

    const double nd0 = std::nextafter(d0, 100.0);
    const double nd1 = std::nextafter(d1, 100.0);
    const double nd2 = std::nextafter(d2, 100.0);

    std::stringstream stream;

    ai::write(stream, d0);
    ai::write(stream, d1);
    ai::write(stream, d2);
    ai::write(stream, nd0);
    ai::write(stream, nd1);
    ai::write(stream, nd2);

    double id0, id1, id2, ind0, ind1, ind2;

    BOOST_CHECK(stream >> id0);
    BOOST_CHECK(stream >> id1);
    BOOST_CHECK(stream >> id2);
    BOOST_CHECK(stream >> ind0);
    BOOST_CHECK(stream >> ind1);
    BOOST_CHECK(stream >> ind2);

    BOOST_CHECK_EQUAL(d0, id0);
    BOOST_CHECK_EQUAL(d1, id1);
    BOOST_CHECK_EQUAL(d2, id2);
    BOOST_CHECK_EQUAL(nd0, ind0);
    BOOST_CHECK_EQUAL(nd1, ind1);
    BOOST_CHECK_EQUAL(nd2, ind2);

    BOOST_CHECK(id0 != ind0);
    BOOST_CHECK(id1 != ind1);
    BOOST_CHECK(id2 != ind2);
}

BOOST_AUTO_TEST_CASE( vectorReadWrite ) {
    auto v = makeRandomVector(5);
    auto bigV = makeRandomVector(6);
    ai::Vector inV(5), inBigV(6);

    std::stringstream stream;

    // We check that the attempted read does not touch the old value.
    inBigV = bigV;

    ai::write(stream, v);
    BOOST_CHECK(!ai::read(stream, inBigV));
    BOOST_CHECK(stream.fail());

    BOOST_CHECK_EQUAL(bigV, inBigV);

    stream.clear();
    stream.str("");

    ai::write(stream, v);
    BOOST_CHECK(ai::read(stream, inV));

    BOOST_CHECK_EQUAL(v, inV);
}

BOOST_AUTO_TEST_CASE( matrix2DReadWrite ) {
    auto m = makeRandomMatrix2D(5, 5);
    auto bigM = makeRandomMatrix2D(5, 6);
    ai::Matrix2D inM(5, 5), inBigM(5, 6);

    std::stringstream stream;

    inBigM = bigM;

    ai::write(stream, m);
    BOOST_CHECK(!ai::read(stream, inBigM));
    BOOST_CHECK(stream.fail());

    BOOST_CHECK_EQUAL(bigM, inBigM);

    stream.clear();
    stream.str("");

    ai::write(stream, m);
    BOOST_CHECK(ai::read(stream, inM));

    BOOST_CHECK_EQUAL(m, inM);
}

BOOST_AUTO_TEST_CASE( sparseMatrix2DReadWrite ) {
    auto m = makeRandomSparseMatrix2D(4, 4);
    auto m2 = makeRandomSparseMatrix2D(4, 4);
    ai::SparseMatrix2D inM = m2;

    std::stringstream stream;

    stream << 26; // Should fail as too many entries;

    BOOST_CHECK(!ai::read(stream, inM));
    BOOST_CHECK(stream.fail());

    BOOST_CHECK(m2.isApprox(inM, 0));

    stream.clear();
    stream.str("");

    stream << "2\n0 0 0.4\n"; // Should fail as too few entries

    BOOST_CHECK(!ai::read(stream, inM));
    BOOST_CHECK(stream.fail());

    BOOST_CHECK(m2.isApprox(inM, 0));

    stream.clear();
    stream.str("");

    ai::write(stream, m);
    BOOST_CHECK(ai::read(stream, inM));

    BOOST_CHECK(m.isApprox(inM, 0));
}

BOOST_AUTO_TEST_CASE( table2DReadWrite ) {
    auto t = makeRandomTable2D(5, 5);
    auto bigT = makeRandomTable2D(5, 6);
    ai::Table2D inT(5, 5), inBigT(5, 6);

    std::stringstream stream;

    inBigT = bigT;

    ai::write(stream, t);
    BOOST_CHECK(!ai::read(stream, inBigT));
    BOOST_CHECK(stream.fail());

    BOOST_CHECK_EQUAL(bigT, inBigT);

    stream.clear();
    stream.str("");

    ai::write(stream, t);
    BOOST_CHECK(ai::read(stream, inT));

    BOOST_CHECK_EQUAL(t, inT);
}

BOOST_AUTO_TEST_CASE( sparseTable2DReadWrite ) {
    auto t = makeRandomSparseTable2D(4, 4);
    auto t2 = makeRandomSparseTable2D(4, 4);
    ai::SparseTable2D inT = t2;

    std::stringstream stream;

    stream << 26; // Should fail as too many entries;

    BOOST_CHECK(!ai::read(stream, inT));
    BOOST_CHECK(stream.fail());

    BOOST_CHECK(t2.isApprox(inT, 0));

    stream.clear();
    stream.str("");

    stream << "2\n0 0 200\n"; // Should fail as too few entries

    BOOST_CHECK(!ai::read(stream, inT));
    BOOST_CHECK(stream.fail());

    BOOST_CHECK(t2.isApprox(inT, 0));

    stream.clear();
    stream.str("");

    ai::write(stream, t);
    BOOST_CHECK(ai::read(stream, inT));

    BOOST_CHECK(t.isApprox(inT, 0));
}

// TODO: Matrix3D, SparseMatrix3D, Table3D, SparseTable3D
