"""Random Instance Generators"""

import numpy as np

import scarf.instance

__all__ = ["gen_random_instance"]


def _id_2_tuple(num_single, idx):
  if idx < num_single:
    return idx
  else:
    couple_idx = idx - num_single
    return (couple_idx // 2, couple_idx % 2)


def gen_random_instance(num_single, num_couple, num_hospital):
  """Generate a random instance."""
  num_applicant = num_single + 2 * num_couple
  single_pref_list = np.argsort(
      np.random.rand(num_single, num_hospital)
  ).tolist()
  num_hospital_pair = (num_hospital + 1) ** 2 - 1
  couple_pref_seed = np.random.rand(num_couple, num_hospital_pair)
  couple_pref_seed += np.array(
      [(pid // (num_hospital + 1) == num_hospital) or 
       (pid % (num_hospital + 1) == num_hospital)
       for pid in range(num_hospital_pair)], dtype=np.float64)
       # plans with unemployment is are ranked lower
  couple_pref_list = np.argsort(couple_pref_seed).tolist()
  wrap = lambda x: x if x < num_hospital else -1
  couple_pref_list = [
      [(wrap(pid // (num_hospital + 1)), 
      	wrap(pid % (num_hospital + 1))) for pid in li]
      for li in couple_pref_list
  ]
  hospital_pref_list = np.argsort(
      np.random.rand(num_hospital, num_applicant)
  ).tolist()
  hospital_pref_list = [
      [_id_2_tuple(num_single, i) for i in li] for li in hospital_pref_list
  ]
  hosp_seat = np.random.randint(
      num_hospital, size=(num_applicant - num_hospital)
  )
  hospital_cap = []
  for h in range(num_hospital):
    hospital_cap.append(int(sum(hosp_seat == h) + 1))
  # print(couple_pref_list)
  
  return scarf.instance.ScarfInstance(
      single_pref_list=single_pref_list,
      couple_pref_list=couple_pref_list,
      hospital_pref_list=hospital_pref_list,
      hospital_cap=hospital_cap
  )

