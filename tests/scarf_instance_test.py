"""Unit tests for scarf package"""

import os
import sys
sys.path.insert(0, os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest

import scarf
from scarf import utils


class TestScarfMethods(unittest.TestCase):
  """docstring for TestScarfMethods"""
  
  def test_pref_list_2_score_list(self):
    hospital_pref_list = [
        [1, (0, 1), 0, (1, 0), (2, 1), (2, 0), (1, 1), (0, 0)],
        [(0, 1), 0, (1, 0), 1, (2, 0), (2, 1), (0, 0), (1, 1)]
    ] 
    single_sc, couple_0_sc, couple_1_sc = scarf.pref_list_2_score_list(
        num_single=2,
        num_couple=3,
        num_hospital=2,
        hospital_pref_list=hospital_pref_list
    )
    single_score = single_sc[0]
    couple_score = list(zip(couple_0_sc[0], couple_1_sc[0]))

    self.assertGreater(single_score[1], couple_score[0][1]) 
    self.assertGreater(couple_score[0][1], single_score[0])
    self.assertGreater(single_score[0], couple_score[1][0])
    self.assertGreater(couple_score[1][0], couple_score[2][1])
    self.assertGreater(couple_score[2][1], couple_score[2][0])
    self.assertGreater(couple_score[2][0], couple_score[1][1])
    self.assertGreater(couple_score[1][1], couple_score[0][0])

  def test_pref_list_2_score_list_idpref(self):
    hospital_pref_list = [
        1, (0, 1), 0, (1, 0), (2, 1), (2, 0), (1, 1), (0, 0)
    ]
    single_sc, couple_0_sc, couple_1_sc = scarf.pref_list_2_score_list(
        num_single=2,
        num_couple=3,
        num_hospital=4,
        hospital_pref_list=hospital_pref_list
    )
    single_score = single_sc[1]
    couple_score = list(zip(couple_0_sc[1], couple_1_sc[1]))

    self.assertGreater(single_score[1], couple_score[0][1]) 
    self.assertGreater(couple_score[0][1], single_score[0])
    self.assertGreater(single_score[0], couple_score[1][0])
    self.assertGreater(couple_score[1][0], couple_score[2][1])
    self.assertGreater(couple_score[2][1], couple_score[2][0])
    self.assertGreater(couple_score[2][0], couple_score[1][1])
    self.assertGreater(couple_score[1][1], couple_score[0][0])


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
            [(1, 1), (0, 0), (0, 2)]
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

  def test_create_pair_list_single(self):
    self.assertListEqual(
        self.S.single_pair_list,
        [(0, 0), (0, 1), (1, 1)]
    )

  def test_create_pair_list_couple(self):
    self.assertListEqual(
        self.S.couple_pair_list,
        [(0, 0, 0), (0, 0, 1), (1, 1, 0), (2, 1, 1), (2, 0, 0), (2, 0, 2)]
    )

  def test_single_ps_utility_matrix(self):
    U_s_ps = self.S._create_single_ps_utility_matrix().toarray()
    self.assertGreater(0, U_s_ps[0][0])
    self.assertGreater(0, U_s_ps[0][1])
    self.assertGreater(0, U_s_ps[1][2])
    self.assertGreater(U_s_ps[0][0], U_s_ps[0][1])

  def test_couple_pc_utility_matrix(self):
    U_c_pc, Vc = self.S._create_couple_pc_utility_matrix()
    U_c_pc = U_c_pc.toarray()
    self.assertGreater(0, U_c_pc[0][0])
    self.assertGreater(0, U_c_pc[0][1])
    self.assertGreater(0, U_c_pc[1][2])
    self.assertGreater(0, U_c_pc[2][3])
    self.assertGreater(0, U_c_pc[2][4])
    self.assertGreater(0, U_c_pc[2][5])

    self.assertGreater(U_c_pc[0][0], U_c_pc[0][1])
    self.assertGreater(U_c_pc[2][3], U_c_pc[2][4])
    self.assertGreater(U_c_pc[2][4], U_c_pc[2][5])

    self.assertGreater(Vc[0], Vc[1])
    self.assertGreater(Vc[3], Vc[4])
    self.assertGreater(Vc[4], Vc[5])

  def test_hospital_ps_utility_matrix(self):
    single_scores = [[-3, -2], [-5, -6]]
    U_h_ps = self.S._create_hospital_ps_utility_matrix(single_scores).toarray()
    # single0:  hosp0 > hosp1, single1: hosp1
    self.assertEqual(-3, U_h_ps[0][0])  # hosp0 rate single 0 
    self.assertEqual(-5, U_h_ps[1][1])  # hosp1 rate single 0
    self.assertEqual(-6, U_h_ps[1][2])  # hosp1 rate single 1

  def test_hospital_0_pc_utility_matrix(self):
    couple_scores = [[-30, -20, -40],  # h0
                     [-50, -60, -70]]  # h1
    Vc = np.array([-1, -2, -1, -1, -2, -3])
    #  [(c0, h0, -), (c0, h0, -), (c1, h1, -), (c2, h1, -), (c2, h0, -), (c2, h0, -)]
    U_h0_pc = self.S._create_hospital_k_pc_utility_matrix(couple_scores, Vc, 0).toarray()
    self.assertEqual(-31, U_h0_pc[0][0]) # hosp0 rate couple 0 (member 0)
    self.assertEqual(-32, U_h0_pc[0][1]) # hosp0 rate couple 0
    self.assertEqual(-61, U_h0_pc[1][2]) # hosp1 rate couple 1
    self.assertEqual(-71, U_h0_pc[1][3]) # hosp1 rate couple 2
    self.assertEqual(-42, U_h0_pc[0][4]) # hosp0 rate couple 2
    self.assertEqual(-43, U_h0_pc[0][5]) # hosp0 rate couple 2

  def test_hospital_1_pc_utility_matrix(self):
    couple_scores = [[-30, -20, -40],  # h0
                     [-50, -60, -70]]  # h1
    Vc = np.array([-1, -2, -1, -1, -2])
    #  [(c0, -, h0), (c0, -, h1), (c1, -, h0), (c2, -, h1), (c2, -, h0), (c2, -, emp)]
    U_h0_pc = self.S._create_hospital_k_pc_utility_matrix(couple_scores, Vc, 1).toarray()
    self.assertEqual(-31, U_h0_pc[0][0]) # hosp0 rate couple 0 (member 1)
    self.assertEqual(-52, U_h0_pc[1][1]) # hosp1 rate couple 0
    self.assertEqual(-21, U_h0_pc[0][2]) # hosp0 rate couple 1
    self.assertEqual(-71, U_h0_pc[1][3]) # hosp1 rate couple 2
    self.assertEqual(-42, U_h0_pc[0][4]) # hosp0 rate couple 2
    self.assertEqual(0, U_h0_pc[0][5])
    self.assertEqual(0, U_h0_pc[1][5])

  def test_utility_matrix(self):
    U = self.S.U.toarray()
    single_pref_list, couple_pref_list, hospital_pref_list = utils.recover_pref_lists(
        num_single=2, num_couple=3, num_hospital=2, U=U,
        pair_list=self.S.single_pair_list + self.S.couple_pair_list
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
    s_pref_list, c_pref_list, h_pref_list_from_U = utils.recover_pref_lists(
        num_single=S.num_single, num_couple=S.num_couple, 
        num_hospital=S.num_hospital, U=S.full_U(),
        pair_list=S.single_pair_list + S.couple_pair_list
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
    for c in range(S.num_couple):
      self.assertListEqual(c_pref_list[c], S.couple_pref_list[c])
    for h in range(S.num_hospital):
      self.assertListEqual(h_pref_list_from_U[h], h_pref_list_from_pref[h])

  def run_solve(self, S):
    basis = scarf.solve(S)
    self.assertTrue(utils.check_stable(S.full_U(), basis))
    self.assertTrue(utils.check_feasible(S.A, basis, S.b))

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

  def test_solve_0(self):
    S = scarf.gen_random_instance(
        num_single=0, num_couple=2, num_hospital=1)
    self.run_solve(S)

  def test_solve_1(self):
    S = scarf.gen_random_instance(
        num_single=2, num_couple=0, num_hospital=1)
    self.run_solve(S)

  def test_solve_2(self):
    S = scarf.gen_random_instance(
        num_single=2, num_couple=2, num_hospital=2)
    self.run_solve(S)

  def test_solve_3(self):
    S = scarf.gen_random_instance(
        num_single=0, num_couple=20, num_hospital=10)
    self.run_solve(S)

  def test_solve_3(self):
    S = scarf.gen_random_instance(
        num_single=20, num_couple=0, num_hospital=10)
    self.run_solve(S)


if __name__ == '__main__':
  unittest.main()
