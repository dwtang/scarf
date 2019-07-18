"""Scarf Algotihm util functions."""

import numpy as np


def check_single_preflist(s, pair_list):
  assert(np.all([len(p) == 2 and p[0] == s for p in pair_list]))
  assert(pair_list[-1] == (s, -1))


def check_couple_preflist(c, pair_list):
  assert(np.all([len(p) == 3 and p[0] == c for p in pair_list]))
  assert(pair_list[-1] == (c, -1, -1))


def check_hospital_preflist(h, pair_list):
  assert(np.all(h in p for p in pair_list))
  assert(pair_list[-1] == (-1, h))


def recover_pref_lists(num_single, num_couple, num_hospital, U,
                       pair_list):
  """Recover preference list from utility matrix."""
  assert(U.shape[0] == num_single + num_couple + num_hospital)
  num_hospital_pair = (num_hospital + 1) ** 2 - 1
  assert(U.shape[1] == len(pair_list))
  # Sort decend by negating the array
  orders_U = np.argsort(-U)
  single_pref_list, couple_pref_list, hospital_pref_list = [], [], []
  for s in range(num_single):
    cols = orders_U[s]
    single_s_pair_list = [pair_list[col] for col in cols if U[s][col] < 0]
    check_single_preflist(s, single_s_pair_list)
    # The last one is (s, num_hospital), remove
    single_pref_list.append([p[1] for p in single_s_pair_list[:-1]])
  for c in range(num_couple):
    cols = orders_U[num_single + c]
    couple_c_pair_list = [pair_list[col] for col in cols 
                          if U[num_single + c][col] < 0]
    check_couple_preflist(c, couple_c_pair_list)
    # The last one is (c, num_hospital, num_hospital), remove
    couple_pref_list.append([p[1:] for p in couple_c_pair_list[:-1]])
  for h in range(num_hospital):
    cols = orders_U[num_single + num_couple + h]
    hospital_h_pref_list = [pair_list[col] for col in cols
                            if U[num_single + num_couple + h][col] < 0]
    check_hospital_preflist(h, hospital_h_pref_list)
    # The last one is (s, num_hospital), remove
    hospital_pref_list.append(hospital_h_pref_list[:-1])
  return single_pref_list, couple_pref_list, hospital_pref_list


def create_hospital_pref_on_pairs(num_hospital, h, one_hospital_pref_list, 
                                  single_pref_list, couple_pref_list):
  """Create hospital's preference on pairs given preference on individuals."""
  pair_pref_list = []
  for i in one_hospital_pref_list:
    if isinstance(i, int):
      if h in single_pref_list[i]:
        pair_pref_list += [(i, h)]
    else:
      c, j = i  # couple c, member j
      # find out if this member is the better one
      member_j_position = one_hospital_pref_list.index((c, j))
      other_member_position = one_hospital_pref_list.index((c, 1 - j))
      is_better_member = member_j_position < other_member_position
      # filter out the pairs related to hospital h and (c, j)
      # where (c, j) is the worst member assigned to couple h in this pair
      def is_relavent_pair(p):
        return p[j] == h and not (p == (h, h) and is_better_member)
      couple_c_pref_list = list(filter(
          is_relavent_pair,
          couple_pref_list[c]
      ))
      pair_pref_list += [(c,) + p for p in couple_c_pref_list]

  return pair_pref_list


def check_stable(U, basis):
  """Check if a basis is ordinal basis for a utility matrix.

  Args:
    U: (m, n) utility matrix.
    basis: a list of m column indices

  Returns:
    True if `basis` is an ordinal basis of `U`
  """
  U_pt = U + np.tile(
    np.linspace(start=0.5, stop=0.0, num=U.shape[1]), 
    (U.shape[0], 1))
  rowmins = np.min(U[:, basis], axis=1, keepdims=True)
  U_dom = U <= rowmins
  return np.all(np.any(U_dom, axis=0))


def check_feasible(A, basis, b):
  """Check if a basis is a feasible basis for a polytope.

  Args:
    A: (m, n) constraint matrix.
    basis: a list of m column indices
    b: the right hand side vector of size (m,)

  Returns:
    True if `basis` is a feasible basis of the polytope `Ax=b, x>=0`.
  """
  if np.linalg.det(A[:, basis]) == 0:
    return False
  else:
    return np.all(np.linalg.solve(A[:, basis], b) >= -1e-6)