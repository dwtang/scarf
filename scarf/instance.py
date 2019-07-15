"""Scarf algorithm library for python3.

Create and solve a stable matching instance with couple using Scarf's algorithm.

Dengwang Tang <dwtang@umich.edu>
"""

import numpy as np
import json
import sys
import time
from scipy import sparse as sp
from scipy import io as sio

import scarf.core

__all__ = [
    "pref_list_2_score_list", "ScarfInstance", "read_excel", 
    "read_json", "gen_random_instance", "solve"
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
      assert(0 <= h0 <= num_hospital)
      assert(0 <= h1 <= num_hospital)
      assert(h0 < num_hospital or h1 < num_hospital)
  
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


def _at(A, list_of_tuples):
  """Accessing nD array A with tuples.

  If list_of_tuples is [(x0, y0), (x1, y1), ..., (xk, yk)], then returns
  [A[x0, y0], A[x1, y1], ..., A[xk, yk]]

  Args:
    A: a numpy ndarray
    list_of_tuples: a list of tuples. Each tuple represent a coordinate

  Returns:
    A 1D array.
  """
  return A[tuple(np.array(list_of_tuples).T)]


def _single_pair_to_column_idx(num_hospital, single_id, hospital_ids):
  """Converts a (single, hospital) pair to column index"""
  return [single_id * num_hospital + i for i in hospital_ids] 


def _couple_pair_to_column_idx(num_hospital, couple_id, hospital_id_pairs):
  """Converts a (couple, hospital0, hospital1) pair to column index"""
  num_hospital_pair = (num_hospital + 1) ** 2 - 1
  return [couple_id * num_hospital_pair + pair[0] * (
    num_hospital + 1) + pair[1] for pair in hospital_id_pairs]


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
  """Create an instance of stable matching problem."""
  def __init__(self, single_pref_list,
               couple_pref_list, hospital_pref_list, hospital_cap):
    """
    Args:
      hospital_cap: a list indicating the number of seat in each hospital.
      single_pref_list: list of list of hospital indices.
      couple_pref_list: list of list of tuple of two hospital indices
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
    self.num_hospital_pair = (self.num_hospital + 1) ** 2 - 1
    self.single_pref_list = single_pref_list
    self.couple_pref_list = couple_pref_list
    self.hospital_pref_list = hospital_pref_list

    self.create_pair_list()  # creates single_pair_list and couple_pair_list
    self.create_constraint_matrix()  # creates self.A
    self.create_utility_matrix()  # creates self.U
    self.create_rhs_vector(hospital_cap)  # creates self.b

  def create_pair_list(self):
    """Creates the list of pairs.

    Creates a list of doctor(s)-hospital(s) pairs (i.e. (s, h), (c, h0, h1))
    which indicates represents the column of constraint and utility matrix.
    """
    self.single_pair_list = []
    for s in range(self.num_single):
      self.single_pair_list += [(s, h) for h in self.single_pref_list[s]]
    self.couple_pair_list = []
    for c in range(self.num_couple):
      self.couple_pair_list += [(c, h0, h1) for h0, h1 
                                in self.couple_pref_list[c]]

  def create_constraint_matrix(self):
    """Create constraint matrix A."""
    AL = np.eye(self.num_single + self.num_couple + self.num_hospital,
                dtype=np.int32)
    A_s_ps = np.repeat(np.eye(self.num_single, dtype=np.int32),
                       self.num_hospital, axis=1)
    A_s_pc = np.zeros(
      (self.num_single, self.num_couple * self.num_hospital_pair),
      dtype=np.int32
    )
    
    A_c_ps = np.zeros(
      (self.num_couple, self.num_single * self.num_hospital),
      dtype=np.int32
    )
    A_c_pc = np.repeat(np.eye(self.num_couple, dtype=np.int32), 
                       self.num_hospital_pair, axis=1)
    
    A_h_ps = np.tile(np.eye(self.num_hospital, dtype=np.int32),
                     (1, self.num_single))
    A_h_pc_first_member = np.concatenate(
        (np.repeat(
            np.eye(self.num_hospital, dtype=np.int32), 
            self.num_hospital + 1, axis=1
        ),
        np.zeros((self.num_hospital, self.num_hospital), dtype=np.int32)),
        axis=1
    )  # [P0, P1, P2, ..., P{num_hospital-1}, 0s]
    A_h_pc_second_member = np.tile(
        np.concatenate((np.eye(self.num_hospital, dtype=np.int32), 
                       np.zeros((self.num_hospital, 1), dtype=np.int32)),
                       axis=1),
        (1, self.num_hospital + 1)
    )  # [[D, 0], [D, 0], ..., [D, 0]]
    A_h_pc_second_member = A_h_pc_second_member[:, :-1]  # remove last column
    A_h_pc_miniblock = A_h_pc_first_member + A_h_pc_second_member
    A_h_pc = np.tile(A_h_pc_miniblock, (1, self.num_couple))

    A_ps = np.concatenate((A_s_ps, A_c_ps, A_h_ps), axis=0)
    A_pc = np.concatenate((A_s_pc, A_c_pc, A_h_pc), axis=0)

    single_indices = []
    for s in range(self.num_single):
      single_indices += _single_pair_to_column_idx(
          self.num_hospital, s, self.single_pref_list[s])
    A_ps = A_ps[:, single_indices]
    
    couple_indices = []
    for c in range(self.num_couple):
      couple_indices += _couple_pair_to_column_idx(
          self.num_hospital, c, self.couple_pref_list[c])
    A_pc = A_pc[:, couple_indices]

    self.A = np.concatenate((AL, A_ps, A_pc), axis=1)

  def _create_single_ps_utility_matrix(self):
    # Construct single - (single, hospital) utility matrix.
    I = [s for s, h in self.single_pair_list]
    J = list(range(len(self.single_pair_list)))
    # For each single, assign score from high to low.
    if self.num_single > 0:
      V = np.concatenate([-np.arange(len(self.single_pref_list[r])) - 1
                          for r in range(self.num_single)])
    else:
      V = np.arange(0)
    return sp.csc_matrix((V, (I, J)), shape=(self.num_single, len(I)),
                         dtype=np.int32)

  def _create_couple_pc_utility_matrix(self):
    """Create couple - (couple, hospitals) utility matrix.

    Returns:
      1. The matrix
      2. A 1-D numpy array of scores couples assigned to pairs, for tie breaking.
    """
    I = [c for c, h0, h1 in self.couple_pair_list]
    J = list(range(len(self.couple_pair_list)))
    # For each couple, assign score from high to low.
    if self.num_couple > 0:
      V = np.concatenate([-np.arange(len(self.couple_pref_list[r])) - 1
                          for r in range(self.num_couple)])
    else:
      V = np.arange(0)
    U_c_pc = sp.csc_matrix((V, (I, J)), shape=(self.num_couple, len(I)),
                           dtype=np.int32)
    return U_c_pc, V + 1

  def _create_hospital_ps_utility_matrix(self, single_scores):
    I = [h for s, h in self.single_pair_list]
    J = list(range(len(self.single_pair_list)))
    # The utility value is the score hospital assigned to single.
    V = [single_scores[h][s] for s, h in self.single_pair_list]

    # Construct the matrix
    return sp.csc_matrix((V, (I, J)), shape=(self.num_hospital, len(I)),
                         dtype=np.int32)

  def _create_hospital_k_pc_utility_matrix(self, couple_scores, V_c_pc, k):
    # Construct hospital - (couple, hospitals) utility matrix separately.
    # First consider first member of couple.
    Ik = [p[k+1] for p in self.couple_pair_list if p[k+1] < self.num_hospital]
    Jk = [j for j in range(len(self.couple_pair_list))
          if self.couple_pair_list[j][k+1] < self.num_hospital]
    # The utility value is the score hospital assigned to member 0 of couple.
    Vk = np.array([couple_scores[p[k+1]][p[0]]
                   for p in self.couple_pair_list 
                   if p[k+1] < self.num_hospital]) + V_c_pc[Jk]
          # adding couple's utility for tie breaking.
    return sp.csc_matrix(
        (Vk, (Ik, Jk)), shape=(self.num_hospital, len(self.couple_pair_list)),
        dtype=np.int32)

  def create_utility_matrix(self):
    """Create utility matrix U"""
    
    U_s_ps = self._create_single_ps_utility_matrix()
    U_c_ps = sp.csc_matrix(
        (self.num_couple, U_s_ps.get_shape()[1]), dtype=np.int32)
    U_c_pc, V_c_pc = self._create_couple_pc_utility_matrix()
    U_s_pc = sp.csc_matrix(
        (self.num_single, U_c_pc.get_shape()[1]), dtype=np.int32)

    single_scores, couple_0_scores, couple_1_scores = pref_list_2_score_list(
        self.num_single, self.num_couple, self.num_hospital,
        self.hospital_pref_list)
    U_h_ps = self._create_hospital_ps_utility_matrix(single_scores)
    U_h0_pc = self._create_hospital_k_pc_utility_matrix(
        couple_0_scores, V_c_pc, 0)
    U_h1_pc = self._create_hospital_k_pc_utility_matrix(
        couple_1_scores, V_c_pc, 1)

    U_h_pc = U_h0_pc.minimum(U_h1_pc)  # Take the worse of the two members

    # Create slack variable utility
    min_u = - (self.num_single + 2 * self.num_couple
        ) * self.num_hospital_pair - 1
    UL = sp.diags(
        [-self.num_hospital - 1 for _ in range(self.num_single)] + 
        [-self.num_hospital_pair - 1 for _ in range(self.num_couple)] + 
        [min_u for _ in range(self.num_hospital)], dtype=np.int32
    )

    self.U = sp.hstack([UL, sp.vstack(
        [
            sp.hstack([U_s_ps, U_s_pc]),
            sp.hstack([U_c_ps, U_c_pc]),
            sp.hstack([U_h_ps, U_h_pc])
        ])
    ])

  def create_rhs_vector(self, hospital_cap):
    self.b = np.concatenate(
        ([1] * (self.num_single + self.num_couple), hospital_cap))

  def savemat(self, filename):
    sio.savemat(
          filename,
          {
              "A": self.A,
              "U": self.U,
              "b": np.array([self.b]).T
          }
    )

  def full_U(self):
    return self.U.toarray()


def read_excel(file):
  """Obtain matching instance from excel"""
  raise NotImplementedError("Developer has been lazy.")


def _element_check(obj):
  if isinstance(obj, list):
    return tuple(obj)
  elif isinstance(obj, int):
    return obj
  else:
    raise TypeError("Wrong!")


def read_json(file):
  """Read instance from a json file of preferences."""
  with open(file) as f:
    all_fields = json.load(f)
  single_pref_list = all_fields["single_pref_list"]
  couple_pref_list = [[_element_check(hp) for hp in li] 
                      for li in all_fields["couple_pref_list"]]
  hospital_pref_list = [[_element_check(i) for i in li]
                        for li in all_fields["hospital_pref_list"]]
  if len(hospital_pref_list) == 1:
    hospital_pref_list = hospital_pref_list[0]
  hospital_cap = all_fields["hospital_cap"]
  return ScarfInstance(
      single_pref_list=single_pref_list,
      couple_pref_list=couple_pref_list,
      hospital_pref_list=hospital_pref_list,
      hospital_cap=hospital_cap
  )


def id_2_tuple(num_single, idx):
  if idx < num_single:
    return idx
  else:
    couple_idx = idx - num_single
    return (couple_idx // 2, couple_idx % 2)


def gen_random_instance(num_single, num_couple, num_hospital, filename=None):
  """Generate a random instance."""
  num_applicant = num_single + 2 * num_couple
  single_pref_list = np.argsort(
      np.random.rand(num_single, num_hospital)
  ).tolist()
  num_hospital_pair = num_hospital ** 2 + 2 * num_hospital
  couple_pref_seed = np.random.rand(num_couple, num_hospital_pair)
  couple_pref_seed += np.array(
      [(pid // (num_hospital + 1) == num_hospital) or 
       (pid % (num_hospital + 1) == num_hospital)
       for pid in range(num_hospital_pair)], dtype=np.float64)
       # plans with unemployment is are ranked lower
  couple_pref_list = np.argsort(couple_pref_seed).tolist()
  couple_pref_list = [
      [(pid // (num_hospital + 1), pid % (num_hospital + 1)) for pid in li]
      for li in couple_pref_list
  ]
  hospital_pref_list = np.argsort(
      np.random.rand(num_hospital, num_applicant)
  ).tolist()
  hospital_pref_list = [
      [id_2_tuple(num_single, i) for i in li] for li in hospital_pref_list
  ]
  hosp_seat = np.random.randint(
      num_hospital, size=(num_applicant - num_hospital)
  )
  hospital_cap = []
  for h in range(num_hospital):
    hospital_cap.append(int(sum(hosp_seat == h) + 1))
  # print(couple_pref_list)
  if filename:
    with open(filename, mode="w") as g:
      json.dump(
          {
              "single_pref_list": single_pref_list,
              "couple_pref_list": couple_pref_list,
              "hospital_pref_list": hospital_pref_list,
              "hospital_cap": hospital_cap
          }, g, indent=4
      )
  return ScarfInstance(
      single_pref_list=single_pref_list,
      couple_pref_list=couple_pref_list,
      hospital_pref_list=hospital_pref_list,
      hospital_cap=hospital_cap
  )


def solve(instance, verbose=False):
  """Solve a stable matching instance with Scarf's algorithm."""
  _, basis = scarf.core.scarf_solve(
      A=instance.A, U=instance.full_U(), b=instance.b, verbose=verbose)
  return basis


if __name__ == '__main__':
  S = read_json(sys.argv[1])
  b, basis = solve(S)