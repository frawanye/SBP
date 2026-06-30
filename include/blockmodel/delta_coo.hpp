/***
 * GPU-instantiable COO delta struct for blockmodel vertex moves.
 *
 * DeltaCOO records the incremental changes to the blockmodel adjacency matrix
 * when a vertex moves from current_block to proposed_block, using three raw
 * parallel arrays (rows, cols, vals) allocated on the device heap.  Every
 * member and method is declared for the GPU target; there is no std::vector,
 * no MapVector, and no host-only stdlib dependency.
 *
 * This struct is a stopgap until the Delta structure can be bypassed entirely.
 * See the detailed migration TODO below.
 *
 * ---------------------------------------------------------------------------
 * TODO: eliminate DeltaCOO (and all device-side allocation) via a merge-walk
 *
 * 1. WHY THIS STRUCT EXISTS AND WHAT TO REMOVE
 *    DeltaCOO buffers per-move changes so delta_mdl / hastings_correction can
 *    read them back via get() / row(i) / col(i) / val(i).  Its three arrays
 *    require a runtime-sized device-heap allocation (new long[capacity]) per
 *    move proposal.  The goal is to never materialise those arrays at all.
 *
 * 2. KEY OBSERVATION
 *    A vertex move perturbs exactly four blockmodel matrix lines:
 *      row[current_block], row[proposed_block],
 *      col[current_block], col[proposed_block].
 *    Every delta entry is a deterministic function of the moving vertex's
 *    out/in neighbor block-assignments and edge weights.  It can be recomputed
 *    on the fly instead of stored.
 *
 * 3. REPLACE THE NEIGHBOR-BLOCK HISTOGRAM (the other device alloc)
 *    Today propose_*_move_gpu builds a MapVector<long> neighbor_blocks.
 *    Instead:
 *      a. Gather the moving vertex's neighbor block-ids into a stack buffer of
 *         size out_deg + in_deg (a small fixed or runtime-bounded array; the
 *         same capacity bound already passed to DeltaCOO).
 *      b. Sort the buffer in place with insertion sort (device-safe, no alloc).
 *      c. Run-length-encode to produce (block, count) pairs — this is the
 *         sorted RLE delta stream for the current_block row/col.
 *    This eliminates MapVector and its device-heap allocation entirely.
 *
 * 4. EXPRESS EACH AFFECTED LINE AS A SORTED STREAM
 *    Do NOT use getrow_sparse / getcol_sparse — those return host-only
 *    MapVector objects and are never called on the GPU.
 *    Instead, read the blockmatrix through its device-side CSR representation
 *    (see include/matrix/csr.hpp), which exposes flat sorted col-index/value
 *    arrays per row, already mapped into the omp target region.
 *    Pair each of the four CSR row/col slices with the sorted RLE delta stream
 *    produced in step 3.
 *
 * 5. MERGE-WALK INSTEAD OF get() / entries() / finalize()
 *    For each of the four affected lines, advance two cursors in lockstep:
 *      - cursor A: the CSR slice for that blockmodel row/col
 *      - cursor B: the sorted RLE delta stream from step 3
 *    At each (r,c) classify the cell as:
 *      (a) changed        — (r,c) present in the delta stream
 *      (b) unchanged but row/col degree shifted — (r,c) only in the CSR slice
 *      (c) new            — (r,c) only in the delta stream
 *    Accumulate the entropy term for that cell immediately into the running dS.
 *    No storage, no sort, no coalesce pass.
 *
 * 6. WHAT THIS DELETES
 *    add, sub, get, finalize (insertion sort + coalesce), the three COO arrays,
 *    the index accessors row(i)/col(i)/val(i), and — critically — ALL
 *    device-side allocation in both DeltaCOO and the neighbor histogram.
 *    Everything then runs over existing CSR arrays plus a couple of small
 *    fixed-size stack buffers.
 *
 * 7. REQUIRED PREREQUISITES
 *    - The blockmatrix must be accessible on the device as sorted flat CSR
 *      arrays (extending include/matrix/csr.hpp's declare-target region).
 *      Explicitly NOT via the host-only _sparse accessors.
 *    - entropy::delta_mdl and hastings_correction must be rewritten to consume
 *      the merge-walk cell classification rather than entries() + get().
 *
 * 8. MIGRATION ORDER
 *    (i)   Land DeltaCOO as the stopgap (done).
 *    (ii)  Add device-callable sorted row/col access to the blockmatrix CSR.
 *    (iii) Rewrite one entropy consumer (e.g. delta_mdl) against the
 *          merge-walk behind a compile-time or runtime flag; validate parity
 *          vs DeltaCOO on the test suite.
 *    (iv)  Extend to all remaining consumers (hastings_correction, etc.).
 *    (v)   Delete DeltaCOO, the neighbor-block histogram MapVector, and
 *          blockmodel_delta_coo once parity is confirmed.
 * ---------------------------------------------------------------------------
 */
#ifndef SBP_BLOCKMODEL_DELTA_COO_HPP
#define SBP_BLOCKMODEL_DELTA_COO_HPP

#include <cstdio>   // printf — device-safe on ROCm/amdclang

#pragma omp declare target

struct DeltaCOO {
private:
    long* _rows;
    long* _cols;
    long* _vals;
    long  _capacity;
    mutable long _nnz;
    mutable bool _finalized;
    long _current_block;
    long _proposed_block;
    long _self_edge_weight;

    /// In-place insertion sort of the three parallel arrays by (row, col)
    /// row-major.  O(n^2) but n is tiny (a few entries per move).
    void sort_entries() const {
        for (long i = 1; i < _nnz; ++i) {
            long r = _rows[i], c = _cols[i], v = _vals[i];
            long j = i - 1;
            while (j >= 0 &&
                   (_rows[j] > r || (_rows[j] == r && _cols[j] > c))) {
                _rows[j + 1] = _rows[j];
                _cols[j + 1] = _cols[j];
                _vals[j + 1] = _vals[j];
                --j;
            }
            _rows[j + 1] = r;
            _cols[j + 1] = c;
            _vals[j + 1] = v;
        }
    }

    /// After sort_entries(), merge adjacent entries with equal (row,col) by
    /// summing their values.  Entries whose coalesced value is zero are
    /// dropped.  Updates _nnz to the compacted count.
    void coalesce() const {
        long write = 0;
        long i = 0;
        while (i < _nnz) {
            long r = _rows[i], c = _cols[i];
            long sum = 0;
            while (i < _nnz && _rows[i] == r && _cols[i] == c) {
                sum += _vals[i];
                ++i;
            }
            if (sum != 0) {
                _rows[write] = r;
                _cols[write] = c;
                _vals[write] = sum;
                ++write;
            }
        }
        _nnz = write;
    }

    /// Sorts and coalesces the COO arrays.  Called lazily before any read.
    void finalize() const {
        if (_finalized) return;
        if (_nnz > 0) {
            sort_entries();
            coalesce();
        }
        _finalized = true;
    }

public:
    /// Constructs an empty DeltaCOO for a move from current_block to
    /// proposed_block.  `capacity` is the maximum number of (row,col,val)
    /// triplets that will be appended before finalize() is called; a safe
    /// upper bound is 2*(out_degree + in_degree) of the moving vertex.
    DeltaCOO(long current_block, long proposed_block, long capacity)
        : _rows(new long[capacity]),
          _cols(new long[capacity]),
          _vals(new long[capacity]),
          _capacity(capacity),
          _nnz(0),
          _finalized(false),
          _current_block(current_block),
          _proposed_block(proposed_block),
          _self_edge_weight(0) {}

    ~DeltaCOO() {
        delete[] _rows;
        delete[] _cols;
        delete[] _vals;
    }

    // -------------------------------------------------------------------------
    // Mutators
    // -------------------------------------------------------------------------

    /// Appends +value as the delta to cell matrix[row][col].
    void add(long row, long col, long value) {
        if (_nnz == _capacity) {
            printf("DeltaCOO::add: capacity %ld exceeded, entry dropped\n",
                   _capacity);
            return;
        }
        _rows[_nnz] = row;
        _cols[_nnz] = col;
        _vals[_nnz] = value;
        ++_nnz;
        _finalized = false;
    }

    /// Appends -value as the delta to cell matrix[row][col].
    void sub(long row, long col, long value) {
        if (_nnz == _capacity) {
            printf("DeltaCOO::sub: capacity %ld exceeded, entry dropped\n",
                   _capacity);
            return;
        }
        _rows[_nnz] = row;
        _cols[_nnz] = col;
        _vals[_nnz] = -value;
        ++_nnz;
        _finalized = false;
    }

    /// Sets the self-edge weight for this move.
    void self_edge_weight(long weight) { _self_edge_weight = weight; }

    // -------------------------------------------------------------------------
    // Scalar accessors
    // -------------------------------------------------------------------------

    long current_block()    const { return _current_block; }
    long proposed_block()   const { return _proposed_block; }
    long self_edge_weight() const { return _self_edge_weight; }
    bool is_coo()           const { return true; }

    // -------------------------------------------------------------------------
    // Read accessors (trigger finalize on first call after any mutation)
    // -------------------------------------------------------------------------

    /// Returns the coalesced number of non-zero entries.
    long nnz() const {
        finalize();
        return _nnz;
    }

    /// Returns the row-index of the i-th entry in the finalized COO.
    long row(long i) const {
        finalize();
        return _rows[i];
    }

    /// Returns the col-index of the i-th entry in the finalized COO.
    long col(long i) const {
        finalize();
        return _cols[i];
    }

    /// Returns the value of the i-th entry in the finalized COO.
    long val(long i) const {
        finalize();
        return _vals[i];
    }

    /// Returns the net delta for matrix[row][col], or 0 if not present.
    long get(long row, long col) const {
        finalize();
        long lo = 0, hi = _nnz;
        while (lo < hi) {
            long mid = (lo + hi) / 2;
            if (_rows[mid] < row || (_rows[mid] == row && _cols[mid] < col))
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < _nnz && _rows[lo] == row && _cols[lo] == col)
            return _vals[lo];
        return 0;
    }

    // -------------------------------------------------------------------------
    // Raw pointer accessors for omp target map (call nnz() / a read accessor
    // first to ensure the arrays are finalized before mapping them to device)
    // -------------------------------------------------------------------------

    const long* rows_data() const { return _rows; }
    const long* cols_data() const { return _cols; }
    const long* vals_data() const { return _vals; }
};

#pragma omp end declare target

#endif // SBP_BLOCKMODEL_DELTA_COO_HPP
