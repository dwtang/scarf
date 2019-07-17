"""Scarf instance input/output."""

import numpy as np
import json
from scipy import io as sio

import scarf.instance

__all__ = ["save_json", "read_json", "save_mat", "read_excel"]


def save_json(ins, filename):
  with open(filename, mode="w") as g:
    json.dump(
        {
            "single_pref_list": ins.single_pref_list,
            "couple_pref_list": ins.couple_pref_list,
            "hospital_pref_list": ins.hospital_pref_list,
            "hospital_cap": ins.hospital_cap
        }, g, indent=4
    )


def _element_check(obj):
  if isinstance(obj, list):
    return tuple(obj)
  elif isinstance(obj, int):
    return obj
  else:
    raise TypeError("Wrong!")


def read_json(filename):
  """Read instance from a json file of preferences."""
  with open(filename) as f:
    all_fields = json.load(f)
  single_pref_list = all_fields["single_pref_list"]
  couple_pref_list = [[_element_check(hp) for hp in li] 
                      for li in all_fields["couple_pref_list"]]
  hospital_pref_list = [[_element_check(i) for i in li]
                        for li in all_fields["hospital_pref_list"]]
  if len(hospital_pref_list) == 1:
    hospital_pref_list = hospital_pref_list[0]
  hospital_cap = all_fields["hospital_cap"]
  return scarf.instance.ScarfInstance(
      single_pref_list=single_pref_list,
      couple_pref_list=couple_pref_list,
      hospital_pref_list=hospital_pref_list,
      hospital_cap=hospital_cap
  )


def save_mat(ins, filename):
  sio.savemat(
        filename,
        {
            "A": ins.A.astype(np.int32),
            "U": ins.U,
            "b": np.array(ins.full_b()).reshape(-1, 1)
        }
  )


def read_excel(file):
  """Obtain matching instance from excel"""
  raise NotImplementedError("Developer has been lazy.")
