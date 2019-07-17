"""Scarf algorithm implementation"""

import numba as nb
import numpy as np
import time

__all__ = ["scarf_solve"]


@nb.njit('int32(int32[:])')
def my_argmin(vec):
  """Argmin function, return's largest index in the case of a tie"""
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


# 'int32(float32[:,:], float32[:], int32[:], int32)'
@nb.njit(nb.i8(nb.float32[:, :], nb.float32[:], 
         nb.types.List(nb.i8, reflected=True), nb.i8), parallel=True)
def cardinal_pivot(A, b, card_basis, pivot):
  """Cardinal pivoting step.

  Introduce `pivot` into the cardinal basis `b` and remove a column j so that 
  `b` is still a cardinal basis.

  Args:
    A: an (m, n) matrix. Must be a non-negative numpy array.
  """
  candrows = np.nonzero(A[:, pivot] > 0)[0]
  ratios = b[candrows] / A[candrows, pivot]
  # find the row with smallest ratio, 
  # so that after operation vector b can stay positive
  row = np.argmin(ratios)
  # transfrom row index to the row index of matrix A
  row = candrows[row]
  col = np.nonzero(np.take(A[row, :], card_basis))[0][0]
  # transfrom col index to the column index of matrix A
  # for col in range(len(card_basis)):
    # if A[row, card_basis[col]] > 0:
      # break
  new_pivot = card_basis[col]

  # In principle, could return new_pivot now can perform the matrix and vector
  # in parallel with ordinal pivot step
  card_basis[col] = pivot

  # Make A[:, pivot] a unit vector (which is 1 at row `row`)
  # Scale row `row` by a factor of 1 / A[row, pivot]
  b[row] = b[row] / A[row, pivot]
  A[row, :] = A[row, :] / A[row, pivot]

  # Subtract a multiple of the row A[row, :] from all other rows
  # not_row = np.concatenate((np.arange(row), np.arange(row+1, len(b))))
  # b[:row] -= A[:row, pivot] * b[row]
  # b[row+1:] -= A[row+1:, pivot] * b[row]
  # b[not_row] -= A[not_row, pivot] * b[row]
  # for r in not_row:  # let numba accelerate the for loop
    # A[r, :] -= A[r, pivot] * A[row, :]
  for r in nb.prange(len(b)):
    if r != row:
      b[r] -= A[r, pivot] * b[row]
      A[r, :] -= A[r, pivot] * A[row, :]
  # A[:row, :] -= np.outer(A[:row, pivot], A[row, :])
  # A[row+1:, :] -= np.outer(A[row+1:, pivot], A[row, :])
  # A[not_row, :] -= np.outer(A[not_row, pivot], A[row, :])

  return new_pivot


# @nb.njit(nb.i8(nb.i4[:,:], nb.types.List(nb.i8, reflected=True), 
    # nb.i4, nb.types.List(nb.i8, reflected=True), nb.b1[:,:]), parallel=True)
# @nb.jit
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
  # ru = rowmin_locations[j]
  # rowmin_locations[j:j+1] = []
  # assert(j == my_argmin(U[ru, basis]))
  pivot_out = ord_basis.pop(j) # remove `pivot` from `basis`
  # ord_basis[j:j+1] = []
  # assert(pivot_out == pivot)
  # Because of removal of `pivot`, row `ru` will have a new row min
  # col_w_2_rowmin = my_argmin_v1(U[ru, :], ord_basis)  # find the new row min location
  col_w_2_rowmin = my_argmin(np.take(U[ru, :], ord_basis))  # find the new row min location
  # returns larger index (smaller actual column index) in case of a tie, 
  # this is desired since we want the abstraction that left 0's are larger than
  # right 0's
  # Now column ord_basis[cn] has two row min, find the original row that has min in
  # this column.
  ri = rowmin_locations[col_w_2_rowmin]  # Our row of interest. 
  # We will use this row to find a new column to ordinal basis
  # row `ru` has a new row min at column ord_basis[cn]
  rowmin_locations[col_w_2_rowmin] = ru  # The row min location is updated
  _update_U_dom_row(U_dom, U, ru, ord_basis[col_w_2_rowmin])

  # new_mins = U[(rowmin_locations, ord_basis)]
  # Find columns that are ONLY dominated via row `ri`
  # not_ri = np.concatenate((np.arange(ri), np.arange(ri+1, U.shape[0])))
  # assert(U_dom[ru, pivot])
  
  # candcols = np.nonzero(~np.any(U_dom[not_ri, :], axis=0))[0]
  # candcols = np.nonzero(np.sum(U_dom[not_ri, :], axis=0) == 0)[0]
  # Only True if all elements in a column are False
  # Ah... np.any with axis param is not supported in nb.njit
  # However I can do this instead!
  # candcols = np.nonzero(
  #     np.sum(U_dom[:ri, :], axis=0) | 
  #     np.sum(U_dom[ri+1:, :], axis=0) == 0)[0] 
  candcols = np.nonzero(
      ~(np.any(U_dom[:ri, :], axis=0) | 
        np.any(U_dom[ri+1:, :], axis=0)))[0] 
      # and I avoided advanced indexing! wowowowowow

  # Among these rows, choose the one with the largest utility
  new_col = np.argmax(np.take(U[ri, :], candcols)) # ri, candcols
  new_pivot = candcols[new_col]

  # In principle, could return here

  # Update basis and book-keeping variables
  j = np.searchsorted(ord_basis, new_pivot)
  ord_basis.insert(j, new_pivot)

  # update U_dom matrix.
  # row `ri` has a new row min at column new_pivot
  rowmin_locations.insert(j, ri)
  _update_U_dom_row(U_dom, U, ri, new_pivot)

  return new_pivot


# @nb.jit
def _scarf_pivot(A, U, b, card_basis, ord_basis, pivot_0, 
                 rowmin_locations, U_dom, verbose=False):
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
  # card_basis
  pivot_steps = 0 
  total_card_time, total_ord_time = 0.0, 0.0
  while True:
    if verbose and pivot_steps % 200 == 0:
      print("current pivot step: #{0}".format(pivot_steps))
    start_time = time.time()
    pivot_1 = cardinal_pivot(A, b, card_basis, pivot_0)
    end_time = time.time()
    duration = end_time - start_time
    total_card_time += duration
    # print("cardinal pivot duration: {0:.6f}ms".format(1e3 * duration))
    # print("{0} replaces {1} in cardinal basis".format(pivot_0, pivot_1))
    if pivot_1 == 0:
      break
    start_time = time.time()
    pivot_0 = ordinal_pivot(U, ord_basis, pivot_1, rowmin_locations, U_dom)
    end_time = time.time()
    duration = end_time - start_time
    # print("ordinal pivot duration: {0:.6f}ms".format(1e3 * duration))
    total_ord_time += duration
    # print("{0} replaces {1} in ordinal basis".format(pivot_0, pivot_1))
    if pivot_0 == 0:
      break
    pivot_steps += 1
  
  print("Pivoted for {0} steps.".format(pivot_steps))
  print("Avg cardinal Pivot: {0:.3f}ms.".format(1e3 * total_card_time / pivot_steps))
  print("Avg Ordinal Pivot: {0:.3f}ms.".format(1e3 * total_ord_time / pivot_steps))
  return card_basis


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
  assert(A.shape[0] < A.shape[1])
  assert(A.shape == U.shape)
  assert(A.shape[0] == len(b))
  # Check that A has a submatrix representing slack variables
  assert(np.all(A[:,:A.shape[0]] == np.eye(A.shape[0], dtype=np.int32)))
  # Check that slack variable of U has lowest utility
  assert(np.all(np.argmin(U, axis=1) == np.arange(A.shape[0])))

  num_rows = len(b)
  A = A.astype(np.float32)
  b = b.astype(np.float32)
  # perturbation for cardinal pivot assumption to hold
  b -= np.linspace(start=1e-6, stop=2e-6, num=num_rows)
  # initialize cardinal and ordinal basis, as well as book keeping variables
  # card_basis = np.arange(num_rows)
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
  
  start_time = time.time()
  basis = _scarf_pivot(A, U, b, card_basis, ord_basis, pivot_0, 
                      rowmin_locations, U_dom, verbose)
  end_time = time.time()
  print("Pivot for {0:.3f}s.".format(end_time - start_time))

  x = np.zeros(shape=(A.shape[1]))
  alloc = np.linalg.solve(A[:, card_basis], b)
  return alloc, basis