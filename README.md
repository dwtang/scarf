# Scarf

Scarf is a python library for solving stable matching problems with couples, where one application is the [National Resident Matching Program](https://en.wikipedia.org/wiki/National_Resident_Matching_Program). The package is named after Hebert Scarf, the inventor of Scarf's lemma and algorithm.

## Features

- Written in python3.
- Solve for a near feasible stable matching given the preference of doctors and hospitals, and hospital capacities.
- Generate random doctor hospital instances.

Accepted input format:
- Python Lists
- JSON

Algorithms:
- Scarf's algorithm with Iterative Rounding [[Nguyen and Vohra 2016]](https://web.ics.purdue.edu/~nguye161/e2sided.pdf)

## Usage

```
  import scarf
  single_pref = [[0, 1, 2], 
                 [1, 0]]
  couple_pref = [[(0, 0), (1, 1), (1, 0), (0, 1), (2, 2)],
                 [(1, 1), (2, 2), (1, -1)]]
  hospital_pref = [0, (0, 1), 1, (1, 0), (0, 0), (1, 1)]
  hospital_cap = [2, 3, 1]
  
  S = scarf.ScarfInstance(single_pref, couple_pref, hospital_pref, hospital_cap)
  sol = scarf.solve(S)
```

## Support

## License

Released under MIT license

```
Copyright (c) 2019 Dengwang Tang <dwtang@umich.edu>
```