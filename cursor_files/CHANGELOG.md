# SBP Change Log

This file documents changes made to the SBP codebase during interactive AI-assisted development sessions.

---

## 2026-06-04 — Dense matrix test coverage

### Summary
Added a dense-matrix (`args.matrix_type = "dense"`) variant for every matrix-dependent unit test in the codebase, plus a dedicated `test/dense_matrix_test.cpp` mirroring the sparse-matrix interface tests. Also fixed a pre-existing bug in `DenseMatrix::getcol_sparseref` / `getrow_sparseref` discovered by the new tests.

### Changes to core files

#### `test/toy_example.hpp`
- Added `std::string forced_matrix_type = ""` protected member to `ToyExample`.
- Modified `ToySetUp(bool transpose)` and `ComplexToySetUp(bool transpose)` to use `forced_matrix_type` when non-empty, overriding the transpose-based matrix type selection. Behavior is fully backward-compatible when `forced_matrix_type` is empty (the default).
- Changed `BlockMergeTest::SetUp()` from implicitly private to `protected:` so that derived dense-fixture classes can call it via `BlockMergeTest::SetUp()`.

#### `include/blockmodel/sparse/dense_matrix.hpp`
- Replaced single `mutable MapVector<long> temp_sparse_vector` with a ring buffer `mutable std::array<MapVector<long>, 4> temp_sparse_vectors` and `mutable size_t temp_vector_idx = 0`.
- **Rationale**: `entropy::delta_mdl` with `SparseEdgeCountUpdates` calls `getrow_sparseref` and `getcol_sparseref` four times and holds all four return-value references simultaneously. With a single shared temp vector, every new call overwrites the previous reference, causing silent data corruption. The ring buffer of 4 ensures each concurrent call gets an independent slot.

#### `src/blockmodel/sparse/dense_matrix.cpp`
- Updated `getcol_sparseref` and `getrow_sparseref` to use the ring-buffer slots instead of `temp_sparse_vector`.

#### `test/dense_matrix_test.cpp` (new file)
- Mirrors `test/dict_matrix_test.cpp`.
- Fixtures: `DenseMatrixTest` and `DenseMatrixComplexTest` (both force `dense`).
- Tests: `NeighborsAreCorrectlyReturned` and `UpdateEdgeCountsIsCorrect` for both fixtures (4 tests).

#### `test/blockmodel_test.cpp`
- Added `BlockmodelDenseTest` and `BlockmodelComplexDenseTest` fixtures.
- Duplicated all 13 matrix-dependent tests (7 from `BlockmodelTest`, 6 from `BlockmodelComplexTest`).

#### `test/block_merge_test.cpp`
- Added `BlockMergeDenseTest` fixture (inherits from `BlockMergeTest`).
- Duplicated `BlockmodelDeltaIsCorrectlyComputed` (1 test).

#### `test/common_test.cpp`
- Added `CommonDenseTest` and `CommonComplexDenseTest` fixtures.
- Duplicated all 4 matrix-dependent tests.

#### `test/entropy_test.cpp`
- Added `EntropyDenseTest` and `BlockMergeEntropyDenseTest` fixtures.
- Duplicated 11 matrix-dependent tests (7 from `EntropyTest`, 4 from `BlockMergeEntropyTest`).
- Excluded: `SetUpWorksCorrectly` (graph-only) and `NullModelMDLv1...LargeGraph` (uses `MockGraph`, no blockmatrix read).

#### `test/finetune_test.cpp`
- Added `FinetuneDenseTest` fixture.
- Duplicated 8 matrix-dependent tests.
- Excluded: `SetUpWorksCorrectly` (graph-only).

#### `test/nonparametric_entropy_test.cpp`
- Added `NonparametricEntropyDenseTest` and `NonparametricBlockMergeEntropyDenseTest` fixtures.
- Duplicated all 10 matrix-dependent tests, **including** the two known-failing degree-corrected tests (`DegreeCorrectedMDLGivesCorrectAnswer` and `DegreeCorrectedDegreeDLGivesCorrectAnswer`).
- Excluded: `SetUpWorksCorrectly` (graph-only) and `EdgesDLGivesCorrectAnswer` (no blockmatrix read).

#### `CMakeLists.txt`
- Added `test/dense_matrix_test.cpp` to the `Test` executable source list.

### Test results

| State | Count |
|---|---|
| Total tests after change | 117 (up from 66) |
| Passing | 113 |
| Failing | 4 |

**Failing tests (all expected):**
1. `NonparametricEntropyTest.DegreeCorrectedMDLGivesCorrectAnswer` — pre-existing
2. `NonparametricEntropyTest.DegreeCorrectedDegreeDLGivesCorrectAnswer` — pre-existing
3. `NonparametricEntropyDenseTest.DegreeCorrectedMDLGivesCorrectAnswer` — expected dense duplicate of #1
4. `NonparametricEntropyDenseTest.DegreeCorrectedDegreeDLGivesCorrectAnswer` — expected dense duplicate of #2

Build and test log: `runs/20260604_153020_dense_test_final.log`

### Dense matrix bug discovered and fixed
The `DenseMatrix::getrow_sparseref` / `getcol_sparseref` methods shared a single `temp_sparse_vector`. This is fine when each reference is used before the next call, but breaks `entropy::delta_mdl(SparseEdgeCountUpdates)` which stores four sparseref return-value references before using any of them. The bug caused `EntropyDenseTest.SparseDeltaMDLGivesCorrectAnswer` to fail until the ring-buffer fix was applied. The fix is backward-compatible and does not affect the sparse matrix types.
