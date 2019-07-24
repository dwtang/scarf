"""Unit tests for scarf package"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import unittest

import scarf
from scarf import utils


class TestScarfClass(unittest.TestCase):
  """docstring for TestScarfClass"""
  def setUp(self):
    self.pref_list = {
        "single": [
            [0, 1],  # single0: hosp0 > hosp1 > unempl 
            [1]  # single1: hosp1 > unempl > hosp0
        ],
        "couple": [
            [(0, 0), (0, 1)],
            [(1, 0)],
            [(1, 1), (0, 0), (0, -1)]
        ],
        "hospital": [
            [1, (0, 1), 0, (1, 0), (2, 1), (2, 0), (1, 1), (0, 0)],
            [(0, 1), 0, (1, 0), 1, (2, 0), (2, 1), (0, 0), (1, 1)]
        ]
    }
    self.S = scarf.ScarfInstance(
        single_pref_list=self.pref_list["single"],
        couple_pref_list=self.pref_list["couple"],
        hospital_pref_list=self.pref_list["hospital"],
        hospital_cap=[2,3]
    )

  def test_pair_list(self):
    self.assertListEqual(
        self.S.pair_list,
        [(0, -1), (1, -1), (0, -1, -1), (1, -1, -1), (2, -1, -1), (-1, 0), (-1, 1), # slack 
         (0, 0), (0, 1), (1, 1),  # single
         (0, 0, 0), (0, 0, 1), (1, 1, 0), (2, 1, 1), (2, 0, 0), (2, 0, -1)]  # couple
    )

  def test_constraint_matrix(self):
    A = self.S.full_A()
    single_0_pair_list = [self.S.pair_list[i] for i in range(A.shape[1])
                          if A[0, i] == 1]
    self.assertListEqual(single_0_pair_list, [(0, -1), (0, 0), (0, 1)])
    single_1_pair_list = [self.S.pair_list[i] for i in range(A.shape[1])
                          if A[1, i] == 1]
    self.assertListEqual(single_1_pair_list, [(1, -1), (1, 1)])
    couple_0_pair_list = [self.S.pair_list[i] for i in range(A.shape[1])
                          if A[2, i] == 1]
    self.assertListEqual(couple_0_pair_list, [(0, -1, -1), (0, 0, 0), (0, 0, 1)])
    couple_1_pair_list = [self.S.pair_list[i] for i in range(A.shape[1])
                          if A[3, i] == 1]
    self.assertListEqual(couple_1_pair_list, [(1, -1, -1), (1, 1, 0)])
    couple_2_pair_list = [self.S.pair_list[i] for i in range(A.shape[1])
                          if A[4, i] == 1]
    self.assertListEqual(couple_2_pair_list, [(2, -1, -1), (2, 1, 1), (2, 0, 0), (2, 0, -1)])
    hospital_0_pair_list_1 = [self.S.pair_list[i] for i in range(A.shape[1])
                              if A[5, i] == 1]
    self.assertListEqual(hospital_0_pair_list_1, [(-1, 0), (0, 0), (0, 0, 1), (1, 1, 0), (2, 0, -1)])
    hospital_0_pair_list_2 = [self.S.pair_list[i] for i in range(A.shape[1])
                              if A[5, i] == 2]
    self.assertListEqual(hospital_0_pair_list_2, [(0, 0, 0), (2, 0, 0)])
    hospital_1_pair_list_1 = [self.S.pair_list[i] for i in range(A.shape[1])
                              if A[6, i] == 1]
    self.assertListEqual(hospital_1_pair_list_1, [(-1, 1), (0, 1), (1, 1), (0, 0, 1), (1, 1, 0)])
    hospital_1_pair_list_2 = [self.S.pair_list[i] for i in range(A.shape[1])
                              if A[6, i] == 2]
    self.assertListEqual(hospital_1_pair_list_2, [(2, 1, 1)])


  def test_utility_matrix(self):
    U = self.S.full_U()
    single_pref_list, couple_pref_list, hospital_pref_list = utils.recover_pref_lists(
        num_single=2, num_couple=3, num_hospital=2, U=U,
        pair_list=self.S.pair_list
    )

    # row minimum should be at slack variable
    minidx = np.argmin(U, axis=1).tolist()
    self.assertListEqual(minidx, list(range(2+3+2)))

    # preference list hospital inferred from U should be consistent
    self.assertListEqual(single_pref_list[0], self.S.single_pref_list[0])
    self.assertListEqual(single_pref_list[1], self.S.single_pref_list[1])
    self.assertListEqual(couple_pref_list[0], self.S.couple_pref_list[0])
    self.assertListEqual(couple_pref_list[1], self.S.couple_pref_list[1])
    self.assertListEqual(couple_pref_list[2], self.S.couple_pref_list[2])

    # preference list of hospital inferred from U should be consistent
    hospital_pref_list_0 = utils.create_hospital_pref_on_pairs(
        num_hospital=2, h=0,
        one_hospital_pref_list=self.pref_list["hospital"][0], 
        single_pref_list=self.S.single_pref_list,
        couple_pref_list=self.S.couple_pref_list
    )
    hospital_pref_list_1 = utils.create_hospital_pref_on_pairs(
        num_hospital=2, h=1,
        one_hospital_pref_list=self.pref_list["hospital"][1], 
        single_pref_list=self.S.single_pref_list,
        couple_pref_list=self.S.couple_pref_list
    )

    self.assertListEqual(hospital_pref_list[0], hospital_pref_list_0)
    self.assertListEqual(hospital_pref_list[1], hospital_pref_list_1)


class TestRandomGenAndSolve(unittest.TestCase):
  def run_test(self, S):
    self.assertTrue(S.A.dtype == np.int8)
    self.assertTrue(S.U.dtype == np.int32)
    s_pref_list, c_pref_list, h_pref_list_from_U = utils.recover_pref_lists(
        num_single=S.num_single, num_couple=S.num_couple, 
        num_hospital=S.num_hospital, U=S.full_U(),
        pair_list=S.pair_list
    )
    h_pref_list_from_pref = [
        utils.create_hospital_pref_on_pairs(
            num_hospital=S.num_hospital, h=h,
            one_hospital_pref_list=S.hospital_pref_list[h], 
            single_pref_list=S.single_pref_list,
            couple_pref_list=S.couple_pref_list
        ) for h in range(S.num_hospital)
    ]
    for s in range(S.num_single):
      self.assertListEqual(s_pref_list[s], S.single_pref_list[s])
      self.assertListEqual([S.hospital_rank_by_single(s, h) for h in S.single_pref_list[s]],
                       list(range(1, len(S.single_pref_list[s]) + 1)))
    for c in range(S.num_couple):
      self.assertListEqual(c_pref_list[c], S.couple_pref_list[c])
      self.assertListEqual([S.hospital_rank_by_couple(c, hs) for hs in S.couple_pref_list[c]],
                           list(range(1, len(S.couple_pref_list[c]) + 1)))
    for h in range(S.num_hospital):
      self.assertListEqual(h_pref_list_from_U[h], h_pref_list_from_pref[h])
      self.assertListEqual([S.doctor_rank_by_hospital(h, s_or_c) for s_or_c in S.hospital_pref_list[h]],
                           list(range(1, len(S.hospital_pref_list[h]) + 1)))

  def run_solve(self, S):
    sol = scarf.solve(S)
    self.assertTrue(utils.check_stable(S.full_U(), sol.basis))
    self.assertTrue(utils.check_feasible(S.full_A(), sol.basis, sol.alloc, S.full_b()))

  def test_small(self):
    S = scarf.gen_random_instance(
        num_single=2, num_couple=0, num_hospital=2)
    self.assertEqual(S.num_single, 2)
    self.assertEqual(S.num_couple, 0)
    self.assertEqual(S.num_hospital, 2)

  def test_0_couple(self):
    S = scarf.gen_random_instance(
        num_single=2, num_couple=0, num_hospital=2)
    
    self.run_test(S)

  def test_0_single(self):
    S = scarf.gen_random_instance(
        num_single=0, num_couple=2, num_hospital=2)
    self.run_test(S)

  def test_1_hosp(self):
    S = scarf.gen_random_instance(
        num_single=0, num_couple=2, num_hospital=1)
    self.run_test(S)

  def test_ihp(self):
    S = scarf.gen_random_instance(
        num_single=10, num_couple=20, num_hospital=10, ihp=True)
    self.run_test(S)

  def test_solve_0(self):
    S = scarf.gen_random_instance(
        num_single=0, num_couple=2, num_hospital=1, ihp=False)
    self.run_solve(S)

  def test_solve_1(self):
    S = scarf.gen_random_instance(
        num_single=2, num_couple=0, num_hospital=1, ihp=False)
    self.run_solve(S)

  def test_solve_2(self):
    S = scarf.gen_random_instance(
        num_single=2, num_couple=2, num_hospital=2, ihp=False)
    self.run_solve(S)

  def test_solve_3(self):
    S = scarf.gen_random_instance(
        num_single=0, num_couple=20, num_hospital=10, ihp=False)
    self.run_solve(S)

  def test_solve_4(self):
    S = scarf.gen_random_instance(
        num_single=20, num_couple=0, num_hospital=10)
    self.run_solve(S)

  def test_solve_5(self):
    S = scarf.gen_random_instance(
        num_single=20, num_couple=10, num_hospital=10, single_pref_len=5, couple_pref_len=10)
    self.run_solve(S)

  def test_solve_ihp(self):
    S = scarf.gen_random_instance(
        num_single=20, num_couple=10, num_hospital=10, single_pref_len=5, couple_pref_len=10, ihp=True)
    self.run_solve(S)

  def run_rounding(self):
    S = scarf.gen_random_instance(
        num_single=10, num_couple=200, num_hospital=15, single_pref_len=10, couple_pref_len=150, ihp=True)
    sol = scarf.solve(S)
    int_sol = scarf.round(sol, S)
    self.assertGreater(sum(sol.act_hospital_cap) + 4 + 1e-6, sum(int_sol.act_hospital_cap))
    self.assertTrue(np.all(int_sol.act_hospital_cap - sol.act_hospital_cap <= 2 + 1e-6))
    self.assertFalse(np.any([(sol[p] < 1e-6) & (int_sol[p] > 1e-6) for p in S.admissible_plans()]))
    self.assertFalse(np.any([(sol[p] > 1 - 1e-6) & (int_sol[p] < 1 - 1e-6) for p in S.admissible_plans()]))

  def test_rounding_0(self):
    self.run_rounding()

  def test_rounding_1(self):
    self.run_rounding()

  def test_rounding_2(self):
    self.run_rounding()


if __name__ == '__main__':
  unittest.main()
