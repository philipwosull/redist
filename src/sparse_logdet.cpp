
#include "sparse_logdet.h"
#include "redist_constants.h" // perf_config

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>


// alias for SparseMatrix
using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// This assumes the matrix is stored as upper triangular one via
// coordinates (ie every entry (i, j, value) has i <= j )
double compute_log_det_from_entries(std::vector<SparseEntry> const &entries,
                                    int const num_rows) {
    // rebuild Eigen triplets here so the header stays free of Eigen
    std::vector<Eigen::Triplet<double, int>> trips;
    trips.reserve(entries.size());
    for (SparseEntry const &e : entries) {
        trips.emplace_back(e.row, e.col, e.value);
    }

    // now make the sparse matrix
    SparseMat sparse_adj_mat(num_rows, num_rows);
    sparse_adj_mat.setFromTriplets(trips.begin(), trips.end());
    sparse_adj_mat.makeCompressed();
    // now factor it

    // 1) Create a sparse Cholesky (LLᵀ) factorization object.
    Eigen::SimplicialLLT<SparseMat> chol;

    // 2) Analyze + factorize the matrix.
    //    - analyzePattern() inspects the sparsity structure (symbolic factorization / ordering)
    //    - factorize() does the numeric factorization
    //    compute() does both in one call (analyzePattern + factorize).
    // Note we assume it is an upper triangular matrix
    chol.compute(sparse_adj_mat.selfadjointView<Eigen::Upper>());

    // 3) Check whether factorization succeeded.
    //    Failure usually means: matrix is not SPD (singular/indefinite) or numerical breakdown.
    if (chol.info() != Eigen::Success) {
        std::ostringstream oss;
        oss << "Sparse Matrix in compute_log_det_from_entries ";
        oss << "was not semi-positive definite" << "\n";
        throw std::runtime_error(oss.str());

        return -INFINITY; // or throw, depending on how you want to handle failure
    }

    // Avoid materializing L: iterate the diagonal directly in the stored sparse factor
    const auto Lview = chol.matrixL();           // TriangularView (in your build)
    const auto &Lmat = Lview.nestedExpression(); // underlying SparseMatrix

    double sumlog = 0.0;
    for (int j = 0; j < num_rows; ++j) {
        bool found = false;

        // Column j scan
        for (SparseMat::InnerIterator it(Lmat, j); it; ++it) {
            if (it.row() == j) { // diagonal entry
                const double d = it.value();
                if (!(d > 0.0)) {
                    std::ostringstream oss;
                    oss << "In compute_log_det_from_entries d was not positive";
                    oss << "\n";
                    throw std::runtime_error(oss.str());
                    return -INFINITY;
                }
                sumlog += std::log(d);
                found = true;
                break;
            }
            // For lower-triangular L, rows are >= j; if we passed j, diagonal is missing
            if (it.row() > j)
                break;
        }

        if (!found) {
            std::ostringstream oss;
            oss << "In compute_log_det_from_entries found was false";
            oss << "\n";
            throw std::runtime_error(oss.str());
            return -INFINITY; // should not happen for successful LLT
        }
    }

    return 2.0 * sumlog;
}


/*
 * LaplacianBuilder
 *
 * The minor has num_rows = num_vertices - 1 rows. Vertex v maps to row v if it
 * comes before the dropped vertex and row v - 1 if after it.
 *
 * Dense mode stores the minor column major in `dense_matrix` with num_rows as
 * the column stride, so the active block is the first num_rows^2 entries.
 * Only the upper triangle (row <= col) is ever written or read.
 *
 * Sparse mode keeps the off diagonal entries as Eigen triplets and the degrees
 * separately, appending the degrees as the diagonal at log_det(). Triplets go
 * straight to setFromTriplets, so nothing is copied.
 */

struct LaplacianBuilder::SparseImpl {
    std::vector<Eigen::Triplet<double, int>> trips; // upper triangle off diagonal entries
    std::vector<double> degrees;                    // diagonal of the minor
    SparseMat matrix;
    Eigen::SimplicialLLT<SparseMat> chol;

    double log_det(int const num_rows);
};

namespace {

// Row of the minor that vertex v maps to. Only meaningful if v != dropped.
inline int minor_row(int const v, int const dropped) { return v < dropped ? v : v - 1; }

// In place Cholesky A = U^T U of the upper triangle of an n x n column major
// matrix with column stride n. Returns log det A = sum of log U_jj^2.
double dense_upper_cholesky_log_det(double *a, int const n) {
    double log_det = 0.0;
    for (int j = 0; j < n; ++j) {
        double *col_j = a + static_cast<std::size_t>(j) * n;
        // off diagonal entries of column j of U
        for (int i = 0; i < j; ++i) {
            double const *col_i = a + static_cast<std::size_t>(i) * n;
            double v = col_j[i];
            for (int k = 0; k < i; ++k) {
                v -= col_i[k] * col_j[k];
            }
            col_j[i] = v / col_i[i];
        }
        // pivot, which is U_jj^2
        double d = col_j[j];
        for (int k = 0; k < j; ++k) {
            d -= col_j[k] * col_j[k];
        }
        if (!(d > 0.0)) {
            std::ostringstream oss;
            oss << "LaplacianBuilder::log_det: dense Laplacian minor is not positive definite "
                << "(pivot " << d << " at row " << j << " of " << n << "). "
                << "This usually means the graph is not connected.\n";
            throw std::runtime_error(oss.str());
        }
        col_j[j] = std::sqrt(d);
        log_det += std::log(d);
    }
    return log_det;
}

} // namespace

double LaplacianBuilder::SparseImpl::log_det(int const num_rows) {
    // degrees go on the diagonal; repeated off diagonal coordinates are summed
    for (int r = 0; r < num_rows; ++r) {
        trips.emplace_back(r, r, degrees[r]);
    }
    matrix.resize(num_rows, num_rows);
    matrix.setFromTriplets(trips.begin(), trips.end());
    matrix.makeCompressed();

    chol.compute(matrix.selfadjointView<Eigen::Upper>());
    if (chol.info() != Eigen::Success) {
        std::ostringstream oss;
        oss << "LaplacianBuilder::log_det: sparse Laplacian minor with " << num_rows
            << " rows is not positive definite. "
            << "This usually means the graph is not connected.\n";
        throw std::runtime_error(oss.str());
    }

    // sum log of L's diagonal without materializing L
    auto const l_view = chol.matrixL();
    auto const &l_mat = l_view.nestedExpression();
    double sum_log = 0.0;
    for (int j = 0; j < num_rows; ++j) {
        bool found = false;
        for (SparseMat::InnerIterator it(l_mat, j); it; ++it) {
            if (it.row() == j) {
                double const d = it.value();
                if (!(d > 0.0)) {
                    std::ostringstream oss;
                    oss << "LaplacianBuilder::log_det: non positive diagonal " << d
                        << " in the sparse Cholesky factor at row " << j << ".\n";
                    throw std::runtime_error(oss.str());
                }
                sum_log += std::log(d);
                found = true;
                break;
            }
            // L is lower triangular, so once past row j the diagonal is missing
            if (it.row() > j) break;
        }
        if (!found) {
            std::ostringstream oss;
            oss << "LaplacianBuilder::log_det: sparse Cholesky factor has no diagonal entry "
                << "at row " << j << ".\n";
            throw std::runtime_error(oss.str());
        }
    }

    return 2.0 * sum_log;
}

LaplacianBuilder::LaplacianBuilder()
    : dense_matrix(static_cast<std::size_t>(dense_max_rows) * dense_max_rows, 0.0) {}

LaplacianBuilder::~LaplacianBuilder() = default;
LaplacianBuilder::LaplacianBuilder(LaplacianBuilder &&) noexcept = default;
LaplacianBuilder &LaplacianBuilder::operator=(LaplacianBuilder &&) noexcept = default;

void LaplacianBuilder::reset(int const num_vertices, int const dropped_vertex) {
    if constexpr (perf_config::supposedly_safe_input_checks) {
        if (num_vertices < 1 || dropped_vertex < 0 || dropped_vertex >= num_vertices) {
            std::ostringstream oss;
            oss << "LaplacianBuilder::reset: need num_vertices >= 1 and "
                << "0 <= dropped_vertex < num_vertices, got num_vertices=" << num_vertices
                << " and dropped_vertex=" << dropped_vertex << ".\n";
            throw std::runtime_error(oss.str());
        }
    }

    this->num_vertices = num_vertices;
    this->dropped_vertex = dropped_vertex;
    num_rows = num_vertices - 1;
    is_dense = num_rows <= dense_max_rows;
    awaiting_log_det = true;

    if (is_dense) {
        // zero the upper triangle of each column of the active block
        for (int j = 0; j < num_rows; ++j) {
            std::fill_n(dense_matrix.begin() + static_cast<std::size_t>(j) * num_rows, j + 1,
                        0.0);
        }
    } else {
        if (!sparse) {
            sparse = std::make_unique<SparseImpl>();
        }
        sparse->trips.clear();
        sparse->degrees.assign(num_rows, 0.0);
    }
}

void LaplacianBuilder::add_edge(int const u, int const v, double const w) {
    if constexpr (perf_config::bounds_checking) {
        if (u < 0 || u >= num_vertices || v < 0 || v >= num_vertices) {
            std::ostringstream oss;
            oss << "LaplacianBuilder::add_edge: edge (" << u << ", " << v
                << ") is out of range for " << num_vertices << " vertices.\n";
            throw std::runtime_error(oss.str());
        }
    }
    if constexpr (perf_config::supposedly_safe_input_checks) {
        if (!(w >= 0.0)) {
            std::ostringstream oss;
            oss << "LaplacianBuilder::add_edge: edge (" << u << ", " << v
                << ") has weight " << w << " but weights must be non-negative.\n";
            throw std::runtime_error(oss.str());
        }
    }

    if (u == v) return;

    bool const u_kept = u != dropped_vertex;
    bool const v_kept = v != dropped_vertex;
    int const u_row = minor_row(u, dropped_vertex);
    int const v_row = minor_row(v, dropped_vertex);

    if (is_dense) {
        double *a = dense_matrix.data();
        std::size_t const stride = num_rows;
        if (u_kept) a[u_row + u_row * stride] += w;
        if (v_kept) a[v_row + v_row * stride] += w;
        if (u_kept && v_kept) {
            int const row = std::min(u_row, v_row);
            int const col = std::max(u_row, v_row);
            a[row + col * stride] -= w;
        }
    } else {
        if (u_kept) sparse->degrees[u_row] += w;
        if (v_kept) sparse->degrees[v_row] += w;
        if (u_kept && v_kept) {
            sparse->trips.emplace_back(std::min(u_row, v_row), std::max(u_row, v_row), -w);
        }
    }
}

double LaplacianBuilder::log_det() {
    if (!awaiting_log_det) {
        throw std::runtime_error(
            "LaplacianBuilder::log_det: called without a reset() since the last log_det().\n");
    }
    awaiting_log_det = false;

    // a single vertex has exactly one spanning tree
    if (num_rows == 0) return 0.0;

    if (is_dense) {
        return dense_upper_cholesky_log_det(dense_matrix.data(), num_rows);
    }
    return sparse->log_det(num_rows);
}