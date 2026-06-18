/***
 * Sparse adjacency matrix in CSR format, for graph (not blockmodel) storage.
 * Designed for GPU offloading via OpenMP target map on MI300A.
 *
 * Storage is raw long* arrays so that read-only methods can be marked
 * `declare target` and called from inside omp target regions without
 * involving std::vector on the device.  All memory management lives in the
 * host-only special members (rule of five).
 */
#ifndef SBP_MATRIX_CSR_HPP
#define SBP_MATRIX_CSR_HPP

#include <algorithm>  // std::copy
#include "typedefs.hpp"

/**
 * CSR-format adjacency matrix for a graph.
 *
 * Three contiguous arrays:
 *   row_ptrs   : size nrows+1 — row_ptrs[v]..row_ptrs[v+1] is the range for vertex v
 *   col_indices: size nedges  — destination vertices in row order
 *   vals       : size nedges  — edge weights (1 for unweighted graphs)
 *
 * Read-only methods are declared target so a const CSR& can be used inside
 * #pragma omp target regions under unified_shared_memory.
 * Constructors / destructor / copy / move are host-only and use new[]/delete[].
 */
class CSR {
  public:
    // -----------------------------------------------------------------------
    // Constructors / destructor / copy / move  (host-only, NOT declare target)
    // -----------------------------------------------------------------------

    CSR() = default;

    /// Build from an adjacency list.
    CSR(const NeighborList &neighbor_list, long num_vertices, long num_edges) {
        nrows  = num_vertices;
        nedges = num_edges;

        row_ptrs    = new long[nrows + 1]();   // zero-initialised
        col_indices = new long[nedges];
        vals        = new long[nedges];

        // Prefix-sum the per-row degrees directly into the row pointers
        // (row_ptrs[0] is already 0 from the zero-initialised allocation).
        for (long v = 0; v < nrows; ++v) {
            row_ptrs[v + 1] = row_ptrs[v] + static_cast<long>(neighbor_list[v].size());
        }

        long pos = 0;
        for (long v = 0; v < nrows; ++v) {
            for (const long neighbor : neighbor_list[v]) {
                col_indices[pos] = neighbor;
                vals[pos]        = 1;
                ++pos;
            }
        }
    }

    ~CSR() {
        delete[] row_ptrs;
        delete[] col_indices;
        delete[] vals;
    }

    CSR(const CSR &other) : nrows(other.nrows), nedges(other.nedges) {
        if (other.row_ptrs) {
            row_ptrs = new long[nrows + 1];
            std::copy(other.row_ptrs, other.row_ptrs + nrows + 1, row_ptrs);
        }
        if (other.col_indices) {
            col_indices = new long[nedges];
            std::copy(other.col_indices, other.col_indices + nedges, col_indices);
        }
        if (other.vals) {
            vals = new long[nedges];
            std::copy(other.vals, other.vals + nedges, vals);
        }
    }

    /// Handles both copy and move assignment via copy-and-swap.
    CSR &operator=(CSR other) noexcept {
        swap(*this, other);
        return *this;
    }

    CSR(CSR &&other) noexcept
        : row_ptrs(other.row_ptrs), col_indices(other.col_indices),
          vals(other.vals), nrows(other.nrows), nedges(other.nedges) {
        other.row_ptrs    = nullptr;
        other.col_indices = nullptr;
        other.vals        = nullptr;
        other.nrows       = 0;
        other.nedges      = 0;
    }

    // -----------------------------------------------------------------------
    // Device-callable read-only API
    // -----------------------------------------------------------------------

#pragma omp begin declare target

    /// Number of rows (vertices).
    long num_rows() const { return nrows; }

    /// Number of stored entries (edges).
    long nnz() const { return nedges; }

    /// Out-degree of vertex v.
    long degree(long v) const {
        return row_ptrs[v + 1] - row_ptrs[v];
    }

    /// View of the neighbors of vertex v (col-index array slice).
    NeighborView neighbors(long v) const {
        return NeighborView(col_indices + row_ptrs[v],
                            row_ptrs[v + 1] - row_ptrs[v]);
    }

    /// Raw pointer to row_ptrs array.
    const long* row_ptrs_data()    const { return row_ptrs; }
    /// Raw pointer to col_indices array.
    const long* col_indices_data() const { return col_indices; }
    /// Raw pointer to vals array.
    const long* vals_data()        const { return vals; }

#pragma omp end declare target

    // -----------------------------------------------------------------------
    // Data members (public for legacy USM host-pointer access if ever needed)
    // -----------------------------------------------------------------------

    long* row_ptrs    = nullptr;   // size nrows+1
    long* col_indices = nullptr;   // size nedges
    long* vals        = nullptr;   // size nedges
    long  nrows       = 0;
    long  nedges      = 0;

  private:
    friend void swap(CSR &a, CSR &b) noexcept {
        using std::swap;
        swap(a.row_ptrs,    b.row_ptrs);
        swap(a.col_indices, b.col_indices);
        swap(a.vals,        b.vals);
        swap(a.nrows,       b.nrows);
        swap(a.nedges,      b.nedges);
    }
};

#endif // SBP_MATRIX_CSR_HPP
