"""
Scarf
=============================================
Solving stable matching problem with couples with Scarf's algorithm.

Example:
---------------------------------------------
	>>> import scarf as scf
	>>> single_pref = [[0, 1, 2], [1, 0]]
	>>> couple_pref = [[(0, 0), (1, 2), (1, 0)], [(2, 1), (1, 2), (1, 3)]]
	>>> hospital_pref = [0, (0, 1), 1, (1, 0), (0, 0), (1, 1)]
	>>> hospital_cap = [3, 4]
	>>> S = scf.ScarfInstance(single_pref, couple_pref, hospital_pref, hospital_cap)
	>>> basis = scf.solve(S)
"""

__author__ = "Dengwang Tang"

from scarf.instance import *
from scarf.io import *
from scarf.random import *