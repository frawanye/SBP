#include <vector>

#include <gtest/gtest.h>

#include "entropy.hpp"
#include "block_merge.hpp"
#include "blockmodel/delta.hpp"

#include "toy_example.hpp"

class BlockMergeDenseTest : public BlockMergeTest {
    void SetUp() override {
        forced_matrix_type = "dense";
        BlockMergeTest::SetUp();
    }
};

TEST_F(BlockMergeTest, BlockmodelDeltaIsCorrectlyComputed) {
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    EXPECT_EQ(delta.entries().size(), 6);
    EXPECT_EQ(delta.get(0,0), -7);
    EXPECT_EQ(delta.get(0,1), -1);
    EXPECT_EQ(delta.get(1,0), -1);
    EXPECT_EQ(delta.get(1,1), 9);
    EXPECT_EQ(delta.get(1,2), 0);
    EXPECT_EQ(delta.get(2,0), -1);
    EXPECT_EQ(delta.get(2,1), 1);
}

TEST_F(BlockMergeDenseTest, BlockmodelDeltaIsCorrectlyComputed) {
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    EXPECT_EQ(delta.entries().size(), 6);
    EXPECT_EQ(delta.get(0,0), -7);
    EXPECT_EQ(delta.get(0,1), -1);
    EXPECT_EQ(delta.get(1,0), -1);
    EXPECT_EQ(delta.get(1,1), 9);
    EXPECT_EQ(delta.get(1,2), 0);
    EXPECT_EQ(delta.get(2,0), -1);
    EXPECT_EQ(delta.get(2,1), 1);
}

// COO-mode variants: same values expected, zero entry (1,2) absent from COO storage but get() still returns 0.

TEST_F(BlockMergeTest, CooBlockmodelDeltaIsCorrectlyComputed) {
    args.coodelta = true;
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    args.coodelta = false;
    EXPECT_EQ(delta.get(0,0), -7);
    EXPECT_EQ(delta.get(0,1), -1);
    EXPECT_EQ(delta.get(1,0), -1);
    EXPECT_EQ(delta.get(1,1), 9);
    EXPECT_EQ(delta.get(1,2), 0);
    EXPECT_EQ(delta.get(2,0), -1);
    EXPECT_EQ(delta.get(2,1), 1);
}

TEST_F(BlockMergeDenseTest, CooBlockmodelDeltaIsCorrectlyComputed) {
    args.coodelta = true;
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    args.coodelta = false;
    EXPECT_EQ(delta.get(0,0), -7);
    EXPECT_EQ(delta.get(0,1), -1);
    EXPECT_EQ(delta.get(1,0), -1);
    EXPECT_EQ(delta.get(1,1), 9);
    EXPECT_EQ(delta.get(1,2), 0);
    EXPECT_EQ(delta.get(2,0), -1);
    EXPECT_EQ(delta.get(2,1), 1);
}
