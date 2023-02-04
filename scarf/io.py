"""Scarf instance input/output."""

import numpy as np
import json
import pickle
from scipy import io as sio

import scarf.instance

__all__ = ["save_json", "load_json", "save_mat", "load_excel",
           "save_pickle", "load_pickle"]


def save_json(ins, filename):
  """Save ScarfInstance to json format.

  Args:
    ins: a `ScarfInstance` object.
    filename: output file name.
  """
  with open(filename, mode="w") as g:
    hospital_pref_list = ins.hospital_pref_list.hospital_pref_list
    if ins.hospital_pref_list.is_ihp: 
      hospital_pref_list = [hospital_pref_list]
    json.dump(
        {
            "single_pref_list": ins.single_pref_list,
            "couple_pref_list": ins.couple_pref_list,
            "hospital_pref_list": hospital_pref_list,
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


def load_json(filename):
  """Read instance from a json file of preferences.

  Args:
    filename: input json file name.
  Returns:
    A `ScarfInstance` object.
  """
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
  """Save part of the instance to MATLAB style .mat file.

  Save the constraint matrix, utility matrix, and right hand side vector into a
  '.mat' file for use in MATLAB.

  Args:
    ins: A `ScarfInstance`.
    filename: output filename with or without '.mat' extension.
  """
  sio.savemat(
        filename,
        {
            "A": ins.A.astype(np.int32),
            "U": ins.U,
            "b": np.array(ins.full_b()).reshape(-1, 1)
        }
  )


def load_excel(filename):
  """Obtain matching instance from excel."""
  raise NotImplementedError("Developer has been lazy.")


def save_pickle(ins, filename):
  """Save ScarfInstance to python's pickle format.

  Args:
    ins: a `ScarfInstance`.
    filename: output file name.
  """
  with open(filename, "wb") as g:
    pickle.dump(
        {
            "single_pref_list": ins.single_pref_list,
            "couple_pref_list": ins.couple_pref_list,
            "hospital_pref_list": ins.hospital_pref_list.hospital_pref_list,
            "hospital_cap": ins.hospital_cap
        }, g
    )


def load_pickle(filename):
  """Read instance from a python pickle file of preferences.

  Warning: As official python3 documentation has suggested, pickle format is NOT
  secure against adversarial attack. Please make sure you trust the source of the
  data file.

  Args:
    filename: json file name.
  Returns:
    A `ScarfInstance` object.
  """
  with open(filename, "rb") as f:
    all_data = pickle.load(f)
  return scarf.instance.ScarfInstance(
      single_pref_list=all_data["single_pref_list"],
      couple_pref_list=all_data["couple_pref_list"],
      hospital_pref_list=all_data["hospital_pref_list"],
      hospital_cap=all_data["hospital_cap"]
  )