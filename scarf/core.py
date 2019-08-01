"""Scarf algorithm implementation"""

import numba as nb
import numpy as np
import scipy.optimize as spopt
from scipy.linalg import blas
import time

__all__ = ["scarf_solve", "iterative_rounding"]


@nb.njit('int32(int32[:])')
def my_argmin(vec):
  """Argmin function, return's largest index in the case of a tie."""
  return len(vec) - 1 - np.argmin(vec[-1:-1-len(vec):-1])


@nb.njit('void(boolean[:,:], int32[:,:], int64, int64)', parallel=True)
def _update_U_dom_row(U_dom, U, r, new_rowmin_col):
  """Update a row of the book-keeping variable U_dom."""
  new_rowmin = U[r, new_rowmin_col]
  if new_rowmin == 0:  # need to ensure left zeros LARGER
    U_dom[r, :new_rowmin_col] = U[r, :new_rowmin_col] < 0
    U_dom[r, new_rowmin_col:] = True  # the one's to the right are always smaller
  else:
    U_dom[r, :] = U[r, :] <= new_rowmin


@nb.njit('void(float64[:,:], float64[:], int64, int64, int64[:])', parallel=True)
def _update_A_b(A, b, row, pivot, oprows):
  """Subtract a multiple of the row A[row, :] from all other rows.

  Implemented with Numba's parallel loop.
  The rows that A[:, pivot] is pre-selected to reduce work.
  """
  for r0 in nb.prange(len(oprows)):
    r = oprows[r0]
    if r != row :
      b[r] -= A[r, pivot] * b[row]
      A[r, :] -= A[r, pivot] * A[row, :]


def _update_A_b_blas(A, b, row, pivot):
  """Subtract a multiple of the row A[row, :] from all other rows.

  Implemented with BLAS routines wrapped by scipy.
  """
  x = np.copy(A[:, pivot])
  x[row] = 0.0  # mask this row
  blas.daxpy(x=x, y=b, a=-b[row])  # y += a * x
  # A += alpha * x * A[row, :]; do it in transpose since A is in C-order.
  blas.dger(alpha=-1.0, x=A[row, :], y=x, a=A.T, overwrite_a=1)


def cardinal_pivot(A, b, card_basis, pivot):
  """Cardinal pivoting step.

  Introduce `pivot` into the cardinal basis `b` and remove a column j so that 
  `b` is still a cardinal basis.

  Args:
    A: an (m, n) matrix. Must be a non-negative numpy array.
  """
  candrows = np.nonzero(A[:, pivot] > 0)[0]
  if candrows.size == 0:
    raise RuntimeError("Cardinal pivot failure.")
  ratios = b[candrows] / A[candrows, pivot]
  # find the row with smallest ratio, 
  # so that after operation vector b can stay positive
  row = np.argmin(ratios)
  # transfrom row index to the row index of matrix A
  row = candrows[row]
  col = np.nonzero(A[row, card_basis])[0][0]
  new_pivot = card_basis[col]

  # In principle, could return new_pivot now can perform the matrix and vector
  # in parallel with ordinal pivot step (parallel is hard to realize on Windows)
  card_basis[col] = pivot

  b[row] /= A[row, pivot]
  A[row, :] /= A[row, pivot]

  # numba prange based implementation seems to be fast for small matrices.
  _update_A_b(A, b, row, pivot, np.nonzero(A[:, pivot])[0])

  # blas based one seems to be faster for large matrix.
  # _update_A_b_blas(A, b, row, pivot)

  # TODO: compare two update functions on a more powerful computer.

  return new_pivot


def ordinal_pivot(U, ord_basis, pivot, rowmin_locations, U_dom):
  """Ordinal pivot step.

  Remove `pivot` from an ordinal basis `basis`, and replace it with another
  column so that `ordinal_basis` is still an ordinal basis.

  Args:
    basis: an increasing array of indices.
    rowmin_locations: array of row index of rowmin location for each column in 
      the basis.
    U_dom: (m, n) boolean array of entry domination.

  Returns:
    The column that was introduced into basis
  """
  j = np.searchsorted(ord_basis, pivot)
  ru = rowmin_locations.pop(j)
  pivot_out = ord_basis.pop(j) # remove `pivot` from `basis`
  col_w_2_rowmin = my_argmin(U[ru, ord_basis])
  # Now column ord_basis[cn] has two row min.
  ri = rowmin_locations[col_w_2_rowmin]  # Our row of interest. 
  # We will use this row to find a new column to ordinal basis
  # row `ru` has a new row min at column ord_basis[cn]
  rowmin_locations[col_w_2_rowmin] = ru  # The row min location is updated
  _update_U_dom_row(U_dom, U, ru, ord_basis[col_w_2_rowmin])

  # Find columns that are ONLY dominated via row `ri`
  candcols = np.nonzero(~(np.any(U_dom[:ri, :], axis=0) | 
                          np.any(U_dom[ri+1:, :], axis=0)))[0]

  # Among these rows, choose the one with the largest utility
  new_col = np.argmax(U[ri, candcols]) # ri, candcols
  new_pivot = candcols[new_col]

  # Could return here for multithread implementation

  # Update basis and book-keeping variables
  j = np.searchsorted(ord_basis, new_pivot)
  ord_basis.insert(j, new_pivot)

  # update U_dom matrix.
  # row `ri` has a new row min at column new_pivot
  rowmin_locations.insert(j, ri)
  _update_U_dom_row(U_dom, U, ri, new_pivot)

  return new_pivot


def _scarf_pivot(A, U, b, verbose=False):
  """Scarf Algorithm Implementation.

  Find a "fractionally stable" extreme point of the polytope {x >= 0 : A x = b}.
  Iteratively apply ordinal and cardinal pivot to obtain a basis that is both
  cardinal and ordinal.

  Args:
    A: Constraint matrix of size (m, n). Must be numpy array with non-negative
       entries.
    U: Utility matrix of size (m, n). Must have numpy array with non-positive
       entries. 
    b: RHS vector of size (n,). Must be numpy array with non-negative entries.

  Returns:
    x: a real-valued vector of length (n,)
    basis: a feasible and stable basis, which is list of column indices of
       length (m,).
  """
  num_rows = len(b)
  A = A.astype(np.float64, order='C')  # for blas functions to work well
  b = b.astype(np.float64)
  # perturbation for cardinal pivot assumption to hold
  # exp(a) is a trancendental number if a is rational!
  # Ain't no polynomial got this solution.
  b += 1e-6 * np.exp(np.linspace(start=0.0, stop=1.0,
                                 num=num_rows, dtype=np.float64))
  # initialize cardinal and ordinal basis, as well as book keeping variables
  card_basis = list(range(num_rows))
  pivot_0 = np.argmax(U[0, num_rows:]) + num_rows
  ord_basis = list(range(1, num_rows)) + [pivot_0]

  # initialize the location (row) of rowmin for each column in the ordinal basis
  rowmin_locations = list(range(1, num_rows)) + [0]
  # initialize the domination matrix
  U_dom = np.concatenate(
    (np.diag([True] * num_rows), 
     np.tile(np.array([[True] + [False] * (num_rows - 1)]).T, 
             A.shape[1] - num_rows)),
    axis=1
  )

  # card_basis
  pivot_steps = 0 
  total_card_time, total_ord_time = 0.0, 0.0
  while True:
    if pivot_steps % 200 == 0 and verbose:
      print("current pivot step: #{0}".format(pivot_steps))
    start_time = time.time()
    pivot_1 = cardinal_pivot(A, b, card_basis, pivot_0)
    end_time = time.time()
    duration = end_time - start_time
    total_card_time += duration
    if pivot_1 == 0:
      break
    start_time = time.time()
    pivot_0 = ordinal_pivot(U, ord_basis, pivot_1, rowmin_locations, U_dom)
    end_time = time.time()
    duration = end_time - start_time
    total_ord_time += duration
    if pivot_0 == 0:
      break
    pivot_steps += 1
  
  return card_basis, pivot_steps, total_card_time, total_ord_time


def scarf_solve(A, U, b, verbose=False):
  """Scarf Algorithm Implementation.

  Find a "fractionally stable" extreme point of the polytope {x >= 0 : A x = b}.
  Iteratively apply ordinal and cardinal pivot to obtain a basis that is both
  cardinal and ordinal.

  Args:
    A: Constraint matrix of size (m, n). Must be numpy array with non-negative
       entries.
    U: Utility matrix of size (m, n). Must have numpy array with non-positive
       entries. 
    b: RHS vector of size (n,). Must be numpy array with non-negative entries.

  Returns:
    x: a real-valued vector of length (n,)
    basis: a list of column indices of length (m,)
  """
  if not A.shape == U.shape:
    raise ValueError("Matrix Dimention mismatch.")
  if not A.shape[0] == len(b):
    raise ValueError("Matrix Vector Dimention mismatch.")
  
  start_time = time.time()
  basis, pivot_steps, total_card_time, total_ord_time = _scarf_pivot(
      A, U, b, verbose)  # A, b are kept same after call.
  end_time = time.time()
  if verbose:
    print("Pivoted for {0} steps in {1:.3f}s.".format(
        pivot_steps, end_time - start_time))
    print("Avg cardinal Pivot: {0:.3f}ms.".format(
        1e3 * total_card_time / pivot_steps))
    print("Avg Ordinal Pivot: {0:.3f}ms.".format(
        1e3 * total_ord_time / pivot_steps))

  try:
    alloc = np.linalg.solve(A[:, basis], b)
  except np.linalg.LinAlgError:
    raise RuntimeError("Scarf algorithm cardinal pivoting failed due to "
                       "floating point precision problem. "
                       "(Yep this is very unlucky...)")
  return alloc, basis, pivot_steps


def iterative_rounding(basis, alloc, A, num_doctor_row, b, tol=1e-6, 
                       verbose=False):
  """Round a feasible and stable fractional solution to a new solution.

  Invented by T. Nguyen and R. Vohra, the Iterative Rounding Method performs a
    rounding procedure on a feasible and stable fractional solution of a stable
    matching problem and returns a integral stable matching which violates the 
    capacity constraint, but the violations are small. This implementation can 
    guarantee that no individual hospital would be overallocated by more than 2
    seats, and the total overallocation will be no more than 9 seats.

  Args:
    basis: The feasible and dominating basis (with no slack variable) to start with.
    alloc: The allocation vector with only the elements in basis.
    A: Doctor constraint matrix, Hospital capacity constraint matrix.
    num_doctor_row: A[:num_doctor_row, :] is the doctor constraint matrix.
    hospital_cap: Hospital capacity vector.
  """

  # basis_np = np.array(basis)
  # non_slack = basis_np >= A.shape[0]

  # A = A[:, basis_np[non_slack]].toarray()
  A = A[:, basis].toarray()
  aggh = np.sum(A[num_doctor_row:, :], axis=0, keepdims=True)
  aggb = np.sum(b[num_doctor_row:])

  A = np.concatenate((A, aggh), axis=0)
  b = np.concatenate((b, [aggb]))
  x = np.round(alloc, 4 - np.log10(tol).astype(np.int64))
 
  active_rows = list(range(num_doctor_row, A.shape[0]))
  agg_still_alive = True
  while True:
    x_is_int = is_int_vec(x, 1e-4 * tol)
    # print("\nIter! x = ", np.round(x[~x_is_int], 10))
    # print("Iter! id = ", np.nonzero(~x_is_int)[0])
    # print("Active rows: ", np.take(A[:, ~x_is_int], active_rows, axis=0))
    b1 = b - rd(np.dot(A[:, x_is_int], x[x_is_int]))
    if np.all(x_is_int):
      # return basis_np[non_slack], rd(x)
      break
    binding = np.dot(A, x) > b1 - tol 
    x_is_frac = ~x_is_int
    candrows = binding[active_rows] & (
        np.dot(A[active_rows, :], x_is_frac) <= 3)
    # print("candrows: ", [active_rows[j] for j in np.nonzero(candrows)[0]])
    if np.any(candrows):
      elim = np.nonzero(candrows)[0][0]
      active_rows.pop(elim)
      # print("# Elim!")
    elif agg_still_alive:
      contain_frac = ~binding[:num_doctor_row] & (
          np.dot(A[:num_doctor_row, ~slack], x_is_frac[~slack]) > 0)
      if np.sum(contain_frac) <= 2:
        active_rows[-1:] = []
        agg_still_alive = False
    else:
      raise RuntimeError("Something Wrong...")

    binding_doctors = np.nonzero(binding[:num_doctor_row])[0]
    userows = np.concatenate(
        (np.nonzero(~binding[:num_doctor_row])[0], active_rows))
    A_eq, b_eq = A[binding_doctors, :], b1[binding_doctors]
    A_ub, b_ub = A[userows, :], b1[userows]
    A_eq, b_eq = A_eq[b_eq > 0, :], b_eq[b_eq > 0]
    A_ub, b_ub = A_ub[b_ub > 0, :], b_ub[b_ub > 0]

    # print("The program to solve: c = ", aggh[0, x_is_frac])
    # print("A_ub = ", A_ub[:, x_is_frac])
    # print("A_eq = ", A_eq[:, x_is_frac])
    # print("b_ub = ", b_ub)
    # print("b_eq = ", b_ub)

    res = spopt.linprog(
      c= -aggh[0, x_is_frac], A_ub=A_ub[:, x_is_frac], b_ub=b_ub,
      A_eq=A_eq[:, x_is_frac], b_eq=b_eq, method="simplex")
    if not res.success:
      raise RuntimeError("Linear program failed!")
    x[x_is_frac] = res.x 
    # print("solution: ", res.x)

  # Intergral solution has been found!
  return rd(x)


def rd(vec):
  return np.round(vec).astype(np.int32)


def is_int(vec, tol=1e-10):
  return np.all(is_int_vec(vec, tol))


def is_int_vec(vec, tol=1e-10):
  return np.abs(vec - np.round(vec)) <= tol
