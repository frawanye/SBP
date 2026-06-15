#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"
#include "matrix/dense_matrix.hpp"

#include "toy_example.hpp"

class DenseMatrixTest : public ToyExample {
    void SetUp() override {
        forced_matrix_type = "dense";
        ToyExample::SetUp();
    }
};

class DenseMatrixComplexTest : public ComplexExample {
    void SetUp() override {
        forced_matrix_type = "dense";
        ComplexExample::SetUp();
    }
};

TEST_F(DenseMatrixTest, NeighborsAreCorrectlyReturned) {
    std::set<long> neighbors = B.blockmatrix()->neighbors(1);
    // neighbors should contain all 3 blocks
    EXPECT_EQ(neighbors.size(), 3);
    for (long i = 0; i < 3; ++i) {
        EXPECT_TRUE(neighbors.find(i) != neighbors.end());
    }
}

TEST_F(DenseMatrixTest, UpdateEdgeCountsIsCorrect) {
    B.blockmatrix()->update_edge_counts(Deltas);
    for (long row = 0; row < B.num_blocks(); ++row) {
        for (long col = 0; col < B.num_blocks(); ++col) {
            long correct_val = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(B.blockmatrix()->get(row, col), correct_val);
        }
    }
}

TEST_F(DenseMatrixComplexTest, NeighborsAreCorrectlyReturned) {
    std::set<long> neighbors = B.blockmatrix()->neighbors(1);
    B.print_blockmatrix();
    std::vector<long> correct_neighbors = { 0, 3, 4, 5 };
    std::cout << "Correct neighbors: " << std::endl;
    utils::print<long>(correct_neighbors);
    std::cout << "Returned neighbors: " << std::endl;
    std::vector<long> n2(neighbors.begin(), neighbors.end());
    utils::print<long>(n2);
    EXPECT_EQ(neighbors.size(), correct_neighbors.size());
    for (const long neighbor : correct_neighbors) {
        EXPECT_TRUE(neighbors.find(neighbor) != neighbors.end());
    }
}

TEST_F(DenseMatrixComplexTest, UpdateEdgeCountsIsCorrect) {
    B.blockmatrix()->update_edge_counts(Deltas);
    for (long row = 0; row < B.num_blocks(); ++row) {
        for (long col = 0; col < B.num_blocks(); ++col) {
            long correct_val = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(B.blockmatrix()->get(row, col), correct_val);
        }
    }
}
