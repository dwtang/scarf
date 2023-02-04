"""
Scarf
=============================================
Solving stable matching problem with couples.

Example:

Suppose that we would like to solve a stable matching problem with 2 single
doctors (s0, s1), 2 couples (c0, c1), and 2 hospitals (h0, h1). We now create
preference lists of all doctors and hospitals, such that:
- 
- for a couple

Say, single doctor s0 ranks hospital h0 > h1 > h2, and single doctor s1 ranks
hospital h1 > h0 > unemployment > h2 (yep, she would rather be unemployed than
going to hospital h2.):

The input preferences list is a list of acceptable options, i.e. 
options that are better than unemployment, from the most preferrable to the least 
preferrable
---------------------------------------------
  >>> single_pref = [[0, 1, 2], 
  ...                [1, 0]]
---------------------------------------------
Different from single doctors, couples makes decision together. Couple c0 (consist of
members c00 and c01) ranks the plan "both go to h0" > "both go to h1" > 
"c00 to h1, c01 to h0" > "c00 to h0, c01 to h1" > "both to h2" > "both unemployed".
Similarly, c1 gives a list, which reads "both to h1" > "both to h2" > 
"c10 to h1, c11 unemplyed" > "both unemployed" > all other plans.

Use -1 to indicate unemployment option. The input preferences list is a list of acceptable
options, i.e. options that are better than "both unemployed", from the most preferrable to 
the least preferrable
---------------------------------------------
  >>> couple_pref = [[(0, 0), (1, 1), (1, 0), (0, 1), (2, 2)],
  ...                [(1, 1), (2, 2), (1, -1)]]
----------------------------------------------
Hospital also has a preference on individual doctors. Here we assume that all three
hospitals use the same preference order: s0 > c01 > s1 > c10 > c00 > c11 > vacancy.
The input preferences list is a list of acceptable doctors, i.e. options that are better than
leaving a seat vacant, from the most preferrable to the least preferrable. Single doctors
are represented by integers. The first member of a couple c is represented with a tuple (c, 0).
The second member of a couple c is represented with a tuple (c, 1).
----------------------------------------------
  >>> hospital_pref = [0, (0, 1), 1, (1, 0), (0, 0), (1, 1)]
----------------------------------------------
Alternatively, if hospitals have different preference over doctors, then we specify
a list of lists satisfying the above format
----------------------------------------------
  >>> hospital_pref = [list0, list1, list2]
----------------------------------------------
Now we specify the capacity of the three hospitals:
----------------------------------------------
  >>> hospital_cap = [2, 3, 1]
----------------------------------------------
Construct the stable matching instance and solve for a potentially fractional stable matching
----------------------------------------------
  >>> import scarf
  >>> S = scarf.create_instance(single_pref, couple_pref, hospital_pref, hospital_cap)
  >>> sol = scarf.solve(S)
----------------------------------------------
In the above example, the above procedure will find an integral stable matching.
However, for many large instances, especially the ones with many couples, Scarf's algorithm
will return a fractional solution. We can then apply Nguyen and Vohra's iterative rounding
algorithm to find an integral matching that may slightly violate the capacity constraint of
the hospitals:
----------------------------------------------
  >>> intsol = scarf.round(sol, S)
----------------------------------------------
Both scarf.solve and scarf.round returns an instance of scarf.ScarfSolution object. 
Please refer to the docstring of scarf.ScarfSolution to see different ways to access the solution.
"""

__author__ = "Dengwang Tang"

from scarf.instance import *
from scarf.io import *
from scarf.random import *