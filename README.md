# Scarf

Scarf is a python library for solving stable matching problems with couples, where one application is the [National Resident Matching Program](https://en.wikipedia.org/wiki/National_Resident_Matching_Program). The package is named after [Herbert Scarf](https://en.wikipedia.org/wiki/Herbert_Scarf), the inventor of Scarf's lemma and algorithm.

## Features

- Written in python3.
- Solve for a near feasible stable matching given the preference of doctors and hospitals, and hospital capacities.
- Generate random doctor hospital instances for simulation purposes.
- Accelerated with Numba. It is fast!

Accepted input format:
- Python Lists
- JSON

Algorithm:
- Scarf's algorithm with Iterative Rounding [[Nguyen and Vohra 2016]](https://web.ics.purdue.edu/~nguye161/e2sided.pdf)

## Installation

```
pip install scarfmatch
```

## Example Usage

```
import scarf

# An example with single doctors {s0, s1}, couples {c0=(c00, c01) c1=(c10, c11)}, and hospitals {h0, h1, h2} 
single_pref = [[0, 1, 2], # s0's preference on hospitals is h0 > h1 > h2 > unemployment
             [1, 0]] # s1's preference on hospitals is h1 > h0 > unemployment > h2
             # list only the hospitals preferred to unemployment 
couple_pref = [[(0, 0), (1, 1), (1, 0), (0, 1), (2, 2)], # c0's joint preference on hospital pairs
             [(1, 1), (2, 2), (1, -1)]] # c1's joint preference on joint plans for both members
             # for example, (1, 0) stands for "member 0 goes to hospital 1 and member 1 goes to hospital 0"
             # "-1" stands for the option of unemployment
             # list only the options preferred to (-1, -1) (i.e. both unemployed)
hospital_pref = [0, (0, 1), 1, (1, 0), (0, 0), (1, 1)] # all three hospitals have the same preference: s0 > c01 > s1 > c10 > c00 > c11 > vacancy
# If all hospital use the same preference, then hospital_pref is one list containing integers and tuples, where integers represent singles and tuples represent members of couples
# To specify different preference lists for different hospitals, please use a list of lists
# In each list, only specify acceptable doctors to the given hospital
hospital_cap = [2, 3, 1] # capacities of each hospital

S = scarf.create_instance(single_pref, couple_pref, hospital_pref, hospital_cap)
sol = scarf.solve(S) # solves for a possibly fractional stable matching
intsol = scarf.round(sol, S) # perform Nguyen and Vohra's IR algorithm to obtain a near feasible stable matching

print(intsol)

```

For more examples involving JSON input please refer to the Jupyter Notebooks in the notebooks folder. For more ways to access the matching result, please refer to the docstring of scarf.ScarfSolution.

## Support

## License

Released under MIT license

```
Copyright (c) 2019-2023 Dengwang Tang <dwtang@umich.edu>
```