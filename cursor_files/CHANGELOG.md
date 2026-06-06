# SBP Change Log

---

## 2026-06-05 — Convert `Graph` adjacency to CSR; add `NeighborView`

### Summary
Converted the `Graph` class's internal adjacency storage from `NeighborList` (vectors of vectors) to a new `CSR` class (`include/matrix/csr.hpp`), providing GPU-mappable contiguous arrays (`row_ptrs`, `col_indices`, `vals`) suitable for future `omp target map` on MI300A.

Key design decisions:
- `include/matrix/gpu_csr.hpp` (the old, incomplete stub) was renamed to `include/matrix/csr.hpp` and the class was renamed from `CSRMatrix` to `CSR`; include guard is now `SBP_MATRIX_CSR_HPP`.
- `CSR` stores three `std::vector<long>` members: `row_ptrs` (V+1), `col_indices` (E), and `vals` (E, filled with `1` for unweighted graphs). `vals` is retained for future weighted-edge support.
- A new `NeighborView` struct (added to `include/typedefs.hpp`) provides a non-owning span (`const long* ptr; long len`) with `begin()/end()/size()/operator[]/empty()/to_vector()`. This replaces `const std::vector<long>&` at per-vertex accessor call sites.
- `Graph` retains staging `NeighborList _out_staging/_in_staging` for incremental construction (`add_edge`). CSR is built at the end of `sort_vertices()` and exposed via `out_csr()`/`in_csr()` for GPU use.
- The whole-list `out_neighbors()`/`in_neighbors()` accessors (returning `const NeighborList&`) were removed; per-vertex `out_neighbors(v)`/`in_neighbors(v)` now return `NeighborView`.

### Files changed
- `include/matrix/gpu_csr.hpp` → deleted; replaced by `include/matrix/csr.hpp` (new `CSR` class)
- `include/typedefs.hpp` — added `NeighborView` struct
- `include/graph.hpp` — removed whole-list accessors; added `out_csr()`/`in_csr()`; added `_csr_ready`, `CSR _out_csr/_in_csr`, renamed staging members
- `src/graph.cpp` — build CSR at end of `sort_vertices()`; `degree`/`degrees` use CSR when ready; staged-only helpers unchanged
- `include/gpu.hpp` — updated include from `matrix/gpu_csr.hpp` to `matrix/csr.hpp`
- `include/distributed/two_hop_blockmodel.hpp` — changed `distribute_2hop_*` and `build_two_hop_blockmodel` signatures from `const NeighborList&` to `const CSR&`
- `src/distributed/two_hop_blockmodel.cpp` — callers now pass `graph.out_csr()`; implementations iterate via `out_csr.neighbors(v)` and `out_csr.num_rows()`
- `src/partition.cpp` — replaced `graph.out_neighbors().size()` with `graph.num_vertices()`
- `src/blockmodel/blockmodel.cpp` — vector-copy site uses `.to_vector()`
- `src/sample.cpp` — binds to `NeighborView` instead of `const std::vector<long>&`
- `include/finetune.hpp` + `src/finetune.cpp` — `edge_weights` signature changed from `const NeighborList&` to `const NeighborView&`; callers pass `graph.out_neighbors(v)` directly
- `src/distributed/dist_finetune.cpp` — same `edge_weights` call-site updates
- `test/finetune_test.cpp`, `test/entropy_test.cpp`, `test/nonparametric_entropy_test.cpp`, `test/sample_test.cpp` — updated to new API

### Test results
113/117 tests pass. The 4 failures are pre-existing (same as before this change):
- `NonparametricEntropyTest.DegreeCorrectedMDLGivesCorrectAnswer`
- `NonparametricEntropyTest.DegreeCorrectedDegreeDLGivesCorrectAnswer`
- `NonparametricEntropyDenseTest.DegreeCorrectedMDLGivesCorrectAnswer`
- `NonparametricEntropyDenseTest.DegreeCorrectedDegreeDLGivesCorrectAnswer`

This file documents changes made to the SBP codebase during interactive AI-assisted development sessions.

---

## 2026-06-05 — Move matrix headers/sources from `blockmodel/sparse/` to `matrix/`

### Summary
Relocated all matrix class headers and sources from `include/blockmodel/sparse/` and `src/blockmodel/sparse/` into new `include/matrix/` and `src/matrix/` directories. This is the first of two steps toward converting the graph's adjacency storage to CSR format for future OpenMP GPU offloading on MI300A.

No logic was changed; this is a pure reorganization. All 113/117 tests pass (the 4 failures are pre-existing: `NonparametricEntropyTest` and `NonparametricEntropyDenseTest` degree-corrected variants, the latter added after the `.cursorrules` baseline was written).

### Files moved
- `include/blockmodel/sparse/{csparse_matrix,dense_matrix,dict_matrix,dict_transpose_matrix,boost_mapped_matrix,gpu_csr,delta,vertex_level_delta}.hpp` → `include/matrix/`
- `src/blockmodel/sparse/{dense_matrix,dict_matrix,dict_transpose_matrix,boost_mapped_matrix,pointer_delta}.cpp` → `src/matrix/`

### Path fixes applied
- `../../utils.hpp` → `../utils.hpp` in `dense_matrix.hpp`, `dict_matrix.hpp`, `dict_transpose_matrix.hpp`, `boost_mapped_matrix.hpp`
- Fixed duplicate include guard: `gpu_csr.hpp` had the same guard (`SBP_BLOCKMODEL_SPARSE_CSR_MATRIX_HPP`) as `include/gpu.hpp`; `gpu_csr.hpp` now uses `SBP_MATRIX_GPU_CSR_HPP`, `gpu.hpp` uses `SBP_GPU_HPP`.
- `pointer_delta.cpp`: `blockmodel/sparse/vertex_level_delta.hpp` → `matrix/vertex_level_delta.hpp`

### Include updates in consuming files
All `#include "blockmodel/sparse/X.hpp"` and `#include "sparse/X.hpp"` references updated to `#include "matrix/X.hpp"` in:
- `include/blockmodel/blockmodel.hpp`
- `include/distributed/two_hop_blockmodel.hpp`
- `include/finetune.hpp`, `include/common.hpp`, `include/block_merge.hpp`
- `include/gpu.hpp`
- `test/dense_matrix_test.cpp`, `test/finetune_test.cpp`, `test/entropy_test.cpp`, `test/nonparametric_entropy_test.cpp`, `test/block_merge_test.cpp`

### Build system
- `CMakeLists.txt`: `include/blockmodel/sparse` → `include/matrix` in `INCLUDE_DIRS`; `src/blockmodel/sparse/*.cpp` → `src/matrix/*.cpp` for the three compiled matrix sources.

---

## 2026-06-04 — Add `--splitrate` command-line argument to TopDownSBP

### Summary
Exposed the TopDownSBP initial-expansion multiplier as a command-line argument `--splitrate` (default 1.25) instead of the previously hardcoded value. This allows tuning the coarseness of block splitting per iteration without recompiling.

### Changes to core files

#### `include/args.hpp`
- Added `float splitrate` member to `Args`.
- Added `TCLAP::ValueArg<float>` for `--splitrate` with default `1.25` and description explaining the effect of different values.
- Added `this->splitrate = _splitrate.getValue()` in the parse block.

#### `src/blockmodel/blockmodel_triplet.cpp`
- Replaced hardcoded `1.25` on line 116 with `args.splitrate` in `TopDownBlockmodelTriplet::get_next_blockmodel()`.

### Background
Experiments at 5k–200k vertices showed that reducing the multiplier from `1.5` (original) to `1.25` consistently improved NMI (+0.02–0.06) and F1 (+0.05–0.12) across all graph sizes and matrix types, while also reducing wall time by ~13–16% at 200k vertices. Making it an argument enables further tuning without recompilation.

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
