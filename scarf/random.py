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


def gen_random_instance(num_single, num_couple, num_hospital,
                        num_additional_seat=0,
                        single_pref_len=0,
                        couple_pref_len=0, ihp=True):
  """Generate a uniform random instance.

  Generate a stable matching instance where single doctor's preference lists are
  chosen uniformly at random, each hospital's ranking of doctors are also chosen
  at random. The couple's preference lists are chosen at random from all lists
  that are unemployment averse.

  Args:
    num_single: int
      Number of single doctors.
    num_couple: int
      Number of coupled doctors.
    num_hospital: int
      Number of hospitals.
    num_additional_seat: int, optional
      Number of total seats will be number of applicants plus num_additional_seat,
      provided that each hospital has at least one seat (Otherwise the hospital
      capacity is 1 for every hospital). Default is 0.
    single_pref_len: int, optional
      Length of single's preference list. Any hospital outside 
      preference list are worse than unemployment option. Default: generate full
      preference list.
    couple_pref_len: int, optional
      Length of couple's preference list. Any hospital pair outside 
      preference list are worse than unemployment option. Default: generate full
      preference list.
    ihp: bool, optional
      If True (default), hospitals will share the same preference on individual
      doctors. Otherwise hospitals will have independent preference lists.

  Returns:
    A `ScarfInstance` object.
  """
  num_applicant = num_single + 2 * num_couple
  single_pref_list = np.argsort(
      np.random.rand(num_single, num_hospital)
  ).tolist()
  if single_pref_len:
    for s in range(num_single):
      single_pref_list[s] = single_pref_list[s][:single_pref_len]
  num_hospital_pair = (num_hospital + 1) ** 2 - 1
  couple_pref_seed = np.random.rand(num_couple, num_hospital_pair)
  couple_pref_seed += np.array(
      [(pid // (num_hospital + 1) == num_hospital) or 
       (pid % (num_hospital + 1) == num_hospital)
       for pid in range(num_hospital_pair)], dtype=np.float64)
       # plans with unemployment is are ranked lower
  couple_pref_list = np.argsort(couple_pref_seed).tolist()
  wrap = lambda x: x if x < num_hospital else -1
  cpl = min(couple_pref_len,
            num_hospital_pair) if couple_pref_len > 0 else num_hospital_pair
  couple_pref_list = [
      [(wrap(pid // (num_hospital + 1)), 
      	wrap(pid % (num_hospital + 1))) for pid in li[:cpl]]
      for li in couple_pref_list
  ]
  if ihp:
    hospital_pref_list = np.argsort(
        np.random.rand(num_applicant)
    ).tolist()
    hospital_pref_list = [
        _id_2_tuple(num_single, i) for i in hospital_pref_list
    ]
  else:
    hospital_pref_list = np.argsort(
        np.random.rand(num_hospital, num_applicant)
    ).tolist()
    hospital_pref_list = [
        [_id_2_tuple(num_single, i) for i in li] for li in hospital_pref_list
    ]
  hosp_seat = np.random.randint(
      num_hospital,
      size=max(num_applicant - num_hospital + num_additional_seat, 0)
  )  # assign each seat randomly to a hospital
  hospital_cap = []
  for h in range(num_hospital):
    hospital_cap.append(int(sum(hosp_seat == h) + 1))
  
  return scarf.instance.ScarfInstance(
      single_pref_list=single_pref_list,
      couple_pref_list=couple_pref_list,
      hospital_pref_list=hospital_pref_list,
      hospital_cap=hospital_cap
  )

