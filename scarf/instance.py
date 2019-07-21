"""Scarf algorithm library for python3.

Create and solve a stable matching instance with couple using Scarf's algorithm.

Dengwang Tang <dwtang@umich.edu>
"""

import numpy as np
from scipy import sparse as sp

import scarf.core

__all__ = [
    "pref_list_2_score_list", "ScarfInstance", "solve"
]

def _assert_unique(li):
  """Check if elements of a list is unique""" 
  assert(len(li) == len(set(li)))


def sanity_check(num_single, num_couple, num_hospital,
                 single_pref_list, couple_pref_list, hospital_pref_list):
  """Sanity check for construction of instances."""
  assert(len(single_pref_list) == num_single)
  assert(len(couple_pref_list) == num_couple)
  
  for li in couple_pref_list:
    _assert_unique(li)
    for h0, h1 in li:
      assert(h0 < num_hospital)
      assert(h1 < num_hospital)
      assert(h0 >= 0 or h1 >= 0)
  
  for li in single_pref_list:
    _assert_unique(li)
    for h in li:
      assert(0 <= h < num_hospital)

  if hospital_pref_list:
    if not isinstance(hospital_pref_list[0], list):
      hospital_pref_list = [hospital_pref_list]
    for li in hospital_pref_list:
      _assert_unique(li)
      for i in li:
        if isinstance(i, int):
          assert(0 <= i < num_single)
        elif isinstance(i, tuple):
          assert(0 <= i[0] < num_couple)
          assert(0 <= i[1] <= 1)


def tuple_2_id(num_single, idx):
  """Transfrom from tuple representation to id.

  If idx is a tuple representation of a member of a couple, transform it to 
  id representation. Otherwise do nothing.

  Raises:
    TypeError if idx is neither an integer or a tuple
  """
  if isinstance(idx, int):
    return idx
  elif isinstance(idx, tuple):
    return num_single + 2 * idx[0] + idx[1]
  else:
    raise TypeError("Cannot recognize idx as single or couple")


def pref_list_2_score_list(num_single, num_couple, num_hospital, hospital_pref_list):
  """Transform one hospital preferences list to a list of scores.

  All returned scores are in range (-num_applicant, 0] * num_hosp_pairs - 1, 
  this allows comparison with 0 in sparse matrices, as well as leaving space
  for tie breaking by perturbing the scores with couple's will. 

  Args:
    num_single: number of single doctors
    num_couple: number of couples
    num_hospital: number of hospitals
    hospital_pref_list: a list of preference lists, where each preference
    list is a list of ints (singles) and tuples (couple, member_id)

  Returns:
    1. Score of singles
    2. Score of first members of couples
    3. Score of second member of couples
  """
  is_identical_pref = not isinstance(hospital_pref_list[0], list)
  num_hospital_pair = (num_hospital + 1) ** 2 - 1
  doctor_scores = np.zeros(
    (num_hospital + 1, num_single + 2 * num_couple), dtype=np.int32)
  for h in range(num_hospital):
    # transform from tuple to idx
    if is_identical_pref:
      hospital_h_pref_list = hospital_pref_list
    else:
      hospital_h_pref_list = hospital_pref_list[h]

    li = list(map(
      lambda x: tuple_2_id(num_single, x), hospital_h_pref_list
    ))
    doctor_scores[h][li] = - np.arange(len(li))

  doctor_scores = doctor_scores * num_hospital_pair - 1
  return (doctor_scores[:num_hospital, :num_single],
          doctor_scores[:, np.arange(
              num_single, num_single + 2 * num_couple, 2)],
          doctor_scores[:, np.arange(
              num_single + 1, num_single + 2 * num_couple, 2)]) 


class ScarfInstance():
  """Stable matching problem instance."""
  def __init__(self, single_pref_list,
               couple_pref_list, hospital_pref_list, hospital_cap):
    """
    Create an instance of stable matching problem.

    Construct a stable matching problem using preference lists and hospital 
    capacity vector.

    Args:
      hospital_cap: a list indicating the number of seat in each hospital.
      single_pref_list: list of list of hospital indices. 
      couple_pref_list: list of list of tuple of two hospital indices. Use -1
        to indicate the option of unemployment.
      hospital_pref_list: 
        - either a list of list of integers and tuples, where integers
          represent the index of a single doctor, and a tuple (c, j) represent
          the j-th member of couple c (j is either 0 or 1)
        - or a list of integers and tuples for identical hospital preference
    """
    
    self.num_single = len(single_pref_list)
    self.num_couple = len(couple_pref_list)
    self.num_hospital = len(hospital_cap)
    # Complete sanity check before proceeding
    sanity_check(self.num_single, self.num_couple, self.num_hospital,
                 single_pref_list, couple_pref_list, hospital_pref_list)
    self.num_applicant = self.num_single + 2 * self.num_couple
    self._nhp = (self.num_hospital + 1) ** 2 - 1

    # store the data
    self.single_pref_list = single_pref_list
    self.couple_pref_list = couple_pref_list
    self.hospital_pref_list = hospital_pref_list
    self.hospital_cap = hospital_cap

    # creates mapping from matrix index to allocation plan
    self._setup_pair_list()
    self.A, self.U = self._create_matrices()

  def __repr__(self):
    s = "<ScarfInstance with {s} singles, {c} couples, and {h} hospitals>".format(
        s=self.num_single, c=self.num_couple, h=self.num_hospital
    )
    return s

  def _setup_pair_list(self):
    """Creates the list of pairs.

    Creates a list of doctor(s)-hospital(s) pairs (i.e. (s, h), (c, h0, h1))
    which indicates represents the column of constraint and utility matrix.
    """
    slack_list = [(s, -1) for s in range(self.num_single)]
    slack_list += [(c, -1, -1) 
                   for c in range(self.num_couple)]
    slack_list += [(-1, h) for h in range(self.num_hospital)]
    single_pair_list = []
    for s in range(self.num_single):
      single_pair_list += [(s, h) for h in self.single_pref_list[s]]
    couple_pair_list = []
    for c in range(self.num_couple):
      couple_pair_list += [(c, h0, h1) for h0, h1 in self.couple_pref_list[c]]
    self.pair_list = slack_list + single_pair_list + couple_pair_list
    self._nsl, self._nps = len(slack_list), len(single_pair_list), 
    self._npc = len(couple_pair_list)

  def slack_list(self):
    return self.pair_list[:self._nsl]

  def single_pair_list(self, j=-1):
    if j < 0:
      return self.pair_list[self._nsl: self._nsl + self._nps]
    else:
      return self.pair_list[self._nsl + j]

  def couple_pair_list(self, j=-1):
    if j < 0:
      return self.pair_list[self._nsl + self._nps:]
    else:
      return self.pair_list[self._nsl + self._nps + j]

  def _create_single_ps_matrices(self):
    # Construct single - (single, hospital) utility matrix.
    I = [s for s, h in self.single_pair_list()]
    J = range(self._nps)
    # For each single, assign score from high to low.
    W = np.ones(self._nps, dtype=np.int8)  # for A
    if self.num_single > 0:
      V = np.concatenate([-np.arange(len(self.single_pref_list[r])) - 1
                          for r in range(self.num_single)])  # for U
    else:
      V = np.arange(0)
    A_s_ps = sp.coo_matrix(
        (W, (I, J)),  shape=(self.num_single, self._nps), dtype=np.int8)
    U_s_ps = sp.coo_matrix(
        (V, (I, J)), shape=(self.num_single, self._nps), dtype=np.int32)
    return A_s_ps, U_s_ps

  def _create_couple_pc_matrices(self):
    """Create couple - (couple, hospitals) utility matrix.

    Returns:
      1. The matrix
      2. A 1-D numpy array of scores couples assigned to pairs, for tie breaking.
    """
    I = [c for c, h0, h1 in self.couple_pair_list()]
    J = range(self._npc)
    W = np.ones(self._npc, dtype=np.int8)
    # For each couple, assign score from high to low.
    if self.num_couple > 0:
      V = np.concatenate([-np.arange(len(self.couple_pref_list[r])) - 1
                          for r in range(self.num_couple)])
    else:
      V = np.arange(0)
    A_c_pc = sp.coo_matrix(
        (W, (I, J)), shape=(self.num_couple, self._npc), dtype=np.int8)
    U_c_pc = sp.coo_matrix(
        (V, (I, J)), shape=(self.num_couple, self._npc), dtype=np.int32)
    return A_c_pc, U_c_pc, V + 1

  def _create_hospital_ps_matrices(self, single_scores):
    I = [h for s, h in self.single_pair_list()]
    J = range(self._nps)
    W = np.ones(self._nps, dtype=np.int8)
    # The utility value is the score hospital assigned to single.
    V = [single_scores[h][s] for s, h in self.single_pair_list()]
    # Construct the matrix
    A_h_ps = sp.coo_matrix(
        (W, (I, J)), shape=(self.num_hospital, self._nps), dtype=np.int8)
    U_h_ps = sp.coo_matrix(
        (V, (I, J)), shape=(self.num_hospital, self._nps), dtype=np.int32)
    return A_h_ps, U_h_ps

  def _create_hospital_k_pc_matrices(self, couple_scores, V_c_pc, k):
    Ik = [p[k+1] for p in self.couple_pair_list() if p[k+1] >= 0]
    Jk = [j for j in range(self._npc)
          if self.couple_pair_list(j)[k+1] >= 0]
    Wk = [1 for j in range(self._npc)
          if self.couple_pair_list(j)[k+1] >= 0]
    # The utility value is the score hospital assigned to member 0 of couple.
    Vk = np.array([couple_scores[p[k+1]][p[0]]
                   for p in self.couple_pair_list()
                   if p[k+1] >= 0]) + V_c_pc[Jk]
          # adding couple's utility for tie breaking.
    A_hk_pc = sp.coo_matrix(
        (Wk, (Ik, Jk)), shape=(self.num_hospital, self._npc), dtype=np.int8)
    U_hk_pc = sp.coo_matrix(
        (Vk, (Ik, Jk)), shape=(self.num_hospital, self._npc), dtype=np.int32)
    return A_hk_pc, U_hk_pc

  def _create_matrices(self):
    """Create constraint and utility matrices for scarf pivoting."""
    
    A_s_ps, U_s_ps = self._create_single_ps_matrices()
    A_s_pc = U_s_pc = sp.coo_matrix(
        (self.num_single, self._npc), dtype=np.int8)
    A_c_ps = U_c_ps = sp.coo_matrix(
        (self.num_couple, self._nps), dtype=np.int8)
    A_c_pc, U_c_pc, V_c_pc = self._create_couple_pc_matrices()

    single_scores, couple_0_scores, couple_1_scores = pref_list_2_score_list(
        self.num_single, self.num_couple, self.num_hospital,
        self.hospital_pref_list)
    A_h_ps, U_h_ps = self._create_hospital_ps_matrices(single_scores)
    A_h0_pc, U_h0_pc = self._create_hospital_k_pc_matrices(
        couple_0_scores, V_c_pc, 0)
    A_h1_pc, U_h1_pc = self._create_hospital_k_pc_matrices(
        couple_1_scores, V_c_pc, 1)

    A_h_pc = A_h0_pc + A_h1_pc
    U_h_pc = U_h0_pc.minimum(U_h1_pc)  # Take the worse of the two members

    AL = sp.eye(self._nsl, dtype=np.int8)
    # Create slack variable utility
    min_u = - self.num_applicant * self._nhp  # the minimum U can achieve
    UL = sp.diags(
        [-self.num_hospital - 1 for _ in range(self.num_single)] + 
        [-self._nhp - 1 for _ in range(self.num_couple)] + 
        [min_u - 1 for _ in range(self.num_hospital)], dtype=np.int32
    )

    A = sp.hstack([AL, sp.vstack(
        [
            sp.hstack([A_s_ps, A_s_pc]),
            sp.hstack([A_c_ps, A_c_pc]),
            sp.hstack([A_h_ps, A_h_pc])
        ])
    ])
    U = sp.hstack([UL, sp.vstack(
        [
            sp.hstack([U_s_ps, U_s_pc]),
            sp.hstack([U_c_ps, U_c_pc]),
            sp.hstack([U_h_ps, U_h_pc])
        ])
    ])
    return A, U

  def full_A(self):
    return self.A.toarray()

  def full_U(self):
    return self.U.toarray()

  def full_b(self):
    return np.concatenate(
        (np.ones(self.num_single + self.num_couple, dtype=np.int32),
         np.array(self.hospital_cap, dtype=np.int32)))


def _is_int(vec, tol=1e-6):
  return np.max(np.abs(vec - np.round(vec))) <= tol


def _s_select(pair_list, s):
  return [p for p in pair_list if len(p) == 2 and p[0] == s and p[1] >= 0]


def _is_c_related(p, c):
  return len(p) == 3 and p[0] == c and (p[1] >= 0 or p[2] >= 0)


def _c_select(pair_list, c):
  return [p for p in pair_list if _is_c_related(p, c)]


def _is_h_related(p, h):
  return p[1] == h or p[-1] == h 


def _h_select(pair_list, h):
  return [p for p in pair_list if _is_h_related(p, h)]


def _get_member(p, h, x):
  if len(p) == 2:
    return [(p[0], x)]
  if p[1] == h and p[2] != h:
    return [((p[0], 0), x)]
  if p[1] != h and p[2] == h:
    return [((p[0], 1), x)]
  if p[1] == h and p[2] == h:
    return [((p[0], 0), x), ((p[0], 1), x)]


class ScarfSolution():
  """Solution of a stable matching instance.

  fields:
    x: a possibly fractional allocation vector, where `x[i]` is the indicator of 
       the allocation plan ins.pair_list[i].
    basis: feasible and ordinal basis as returned by scarf algorithm.
    num_pivots: number of pivots performed.
    is_int: True if the solution is integral.
    hospital_slack: number of empty seats in hospitals after allocation.
  """
  def __init__(self, S, alloc, basis, num_pivots=-1, tol=1e-6):
    """
    Args:
      S: a `ScarfInstance` object.
      alloc: allocation vector.
      basis: indices.
      num_pivots: number of pivots in the algorithm.
      tol: tolerance for integral testing.
    """
    # For linear algebra ensusiasts
    self.x = sp.csc_matrix((alloc, (basis, np.zeros(len(basis)))), 
                           shape=(S.A.shape[1], 1), dtype=np.float64)
    self.single_slack = self.x[:S.num_single].T.toarray()[0].tolist()
    self.couple_slack = self.x[
        S.num_single:S.num_single + S.num_couple].T.toarray()[0].tolist()
    self.hospital_slack = self.x[
        S.num_single + S.num_couple:
        S.num_single + S.num_couple + S.num_hospital].T.toarray()[0].tolist()
    self.basis = basis
    self.num_pivots = num_pivots
    self.is_int = _is_int(alloc, tol)
    # For solution lookup
    self._num_hospital = S.num_hospital
    self._solution_map = {S.pair_list[basis[i]]: alloc[i] 
                          for i in range(len(basis)) if alloc[i] > tol}
    _pair_list = self._solution_map.keys()
    self.single_allocations = [
        _s_select(_pair_list, s) for s in range(S.num_single)] 
    self.couple_allocations = [
        _c_select(_pair_list, c) for c in range(S.num_couple)]
    self.hospital_allocations = [
        _h_select(_pair_list, h) for h in range(S.num_hospital)]

  def __repr__(self):
    s = ["<ScarfSolution of {s} singles, {c} couples, {h} hospitals".format(
        s=len(self.single_allocations), c=len(self.couple_allocations),
        h=self._num_hospital
    )] 
    s += ["\twith {0} solution>".format(
        "integral" if self.is_int else "NON integral")
    ]
    return "\n".join(s)

  def __getitem__(self, s):
    """Get matched hospital/doctor.

    Args:
      s: either a tuple indicating a doctor(s)-hospital(s) pair, or a string
         indicating a single (e.g. "s10") or a couple ("c2") or a hospital
         (e.g. "h12")

    Returns:
      a number between 0 and 1 (allocation fraction) if `s` is a doctor(s)-hospital(s)
         pair; a dictionary from hospital/hospitals/doctor to allocation fraction
         if `s` is a string indicating single/couple/hospital.
    """
    if isinstance(s, tuple):
      if s in self._solution_map:
        return self._solution_map[s]
      else:
        return 0.0
    elif isinstance(s, str):
      if s.startswith("s"):
        return self.get_single_allocation(int(s[1:]))
      elif s.startswith("c"):
        return self.get_couple_allocation(int(s[1:]))
      elif s.startswith("h"):
        return self.get_hospital_allocation(int(s[1:]))
      else:
        raise TypeError("Unrecognized index.")

  def get_single_allocation(self, s):
    return {p[1]: self._solution_map[p] for p in self.single_allocations[s]}

  def get_couple_allocation(self, c):
    return {p[1:]: self._solution_map[p] for p in self.couple_allocations[c]}

  def get_hospital_allocation(self, h):
    ll = []
    for p in self.hospital_allocations[h]:
      ll += _get_member(p, h, self._solution_map[p])
    return dict(ll)


def solve(ins, verbose=False):
  """Solve a stable matching instance with Scarf's algorithm.

  Args:
    ins: a `ScarfInstance` object.
    verbose: Extra information will be printed during running the algorithm if
       set to be true.

  Returns:
    sol: a `ScarfSolution` object.
    fields:
      x: a possibly fractional allocation vector, where `x[i]` is the indicator of 
         the allocation plan ins.pair_list[i].
      basis: feasible and ordinal basis as returned by scarf algorithm.
      num_pivots: number of pivots performed.
      is_int: True if the solution is integral.
      hospital_slack: number of empty seats in hospitals after allocation.
    `sol` is also indexable.
  """
  alloc, basis, num_pivots = scarf.core.scarf_solve(
      A=ins.full_A(),
      U=ins.full_U(),
      b=ins.full_b(),
      verbose=verbose
  )
  return ScarfSolution(ins, alloc, basis, num_pivots)
