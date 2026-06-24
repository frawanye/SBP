/***
 * Sparse adjacency matrix in CSR format, for graph (not blockmodel) storage.
 * Designed for GPU offloading via OpenMP target map on MI300A.
 *
 * CSR is a trivially-copyable minimal CSR Matrix.  All declare-target methods
 * use only raw pointer arithmetic so they compile cleanly for GPU targets.
 * Memory management (new[]/delete[]) lives exclusively in Graph.
 */
#ifndef SBP_MATRIX_CSR_HPP
#define SBP_MATRIX_CSR_HPP

#include <type_traits>
#include "typedefs.hpp"

/**
 * CSR-format adjacency matrix for a graph.
 *
 * Three contiguous arrays (owned by Graph):
 *   row_ptrs   : size nrows+1 — row_ptrs[v]..row_ptrs[v+1] is the range for vertex v
 *   col_indices: size nedges  — destination vertices in row order
 *   vals       : size nedges  — edge weights (1 for unweighted graphs)
 *
 * CSR is a trivially-copyable minimal CSR matrix so it can be mapped into omp target regions
 * without -Wopenmp-mapping warnings.  It carries no ownership; the Graph that
 * builds it is responsible for the lifetime of the arrays.
 */
#pragma omp begin declare target
struct CSR {
    long* row_ptrs    = nullptr;   // size nrows+1
    long* col_indices = nullptr;   // size nedges
    long* vals        = nullptr;   // size nedges
    long  nrows       = 0;
    long  nedges      = 0;

    /// Number of rows (vertices).
    long num_rows() const { return nrows; }

    /// Number of stored entries (edges).
    long nnz() const { return nedges; }

    /// Out-degree of vertex v.
    long degree(long v) const { return row_ptrs[v + 1] - row_ptrs[v]; }

    /// View of the neighbors of vertex v (col-index and weight slices).
    NeighborView neighbors(long v) const {
        return NeighborView(col_indices + row_ptrs[v], vals + row_ptrs[v], row_ptrs[v + 1] - row_ptrs[v]);
    }

    

    /// Raw pointer to row_ptrs array.
    const long* row_ptrs_data()    const { return row_ptrs; }
    /// Raw pointer to col_indices array.
    const long* col_indices_data() const { return col_indices; }
    /// Raw pointer to vals array.
    const long* vals_data()        const { return vals; }
};
#pragma omp end declare target

static_assert(std::is_trivially_copyable_v<CSR>, "CSR must stay trivially copyable for GPU mapping");

#endif // SBP_MATRIX_CSR_HPP
