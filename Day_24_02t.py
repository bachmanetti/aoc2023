# This absolutely atrocious code attemps to solve AOC 2023 day 24 parts 1 and 2
# For Part 2, it does this by using a small random set of the whole
# Then we assume that if we were to add the inverse of the rocks velocity to the hail, we eventually will have
# all hailstone paths intersecting on a single point (the rocks source)
# So now we test various combinations of velocities to determine the best place to search for the "lowest"
# average distance between all hailstone paths.
# We keep doing this until it can't find a lower value in its search area

import re
import math
import numpy as np
import itertools
import random
from aocd import get_data
from aocd.models import Puzzle

aoc_day = 24

puzzle = Puzzle(year=2023, day=aoc_day)
input_data = puzzle.input_data.split("\n")

dim_2d = {
    'xy': (0, 1),
    'yz': (1, 2),
    'xz': (0, 2)
}

test_min = 200000000000000
test_max = 400000000000000

example_data = False

# Get Example Data
if example_data:
    input_data = puzzle.examples[0].input_data.split("\n")
    test_min = 7
    test_max = 27

hail_paths = []
for line in input_data:
    strparse = re.findall(r'-?\d+', line)
    strparse = [int(a) for a in strparse]
    hail_paths.append(strparse)


def calc_intercepts(hail1, hail2):
    intercepts = {}
    for key, dim in dim_2d.items():
        L1 = {'a': -hail1[dim[1] + 3],
              'b': hail1[dim[0] + 3],
              'c': hail1[dim[1] + 3] * hail1[dim[0]] - hail1[dim[0] + 3] * hail1[dim[1]]}
        L2 = {'a': -hail2[dim[1] + 3],
              'b': hail2[dim[0] + 3],
              'c': hail2[dim[1] + 3] * hail2[dim[0]] - hail2[dim[0] + 3] * hail2[dim[1]]}
        if (L1['a'] * L2['b'] - L2['a'] * L1['b']) == 0:
            intercepts[key] = False
        else:
            D0 = (L1['b'] * L2['c'] - L2['b'] * L1['c']) / (L1['a'] * L2['b'] - L2['a'] * L1['b'])
            D1 = (L1['c'] * L2['a'] - L2['c'] * L1['a']) / (L1['a'] * L2['b'] - L2['a'] * L1['b'])
            test1 = (D0 - hail1[dim[0]]) / hail1[dim[0] + 3] if hail1[dim[0] + 3] != 0 else (
                        D1 - hail1[dim[1]]) / hail1[dim[1] + 3]
            test2 = (D0 - hail2[dim[0]]) / hail2[dim[0] + 3] if hail2[dim[0] + 3] != 0 else (
                        D1 - hail2[dim[1]]) / hail2[dim[1] + 3]
            if test1 > 0 and test2 > 0:
                intercepts[key] = (D0, D1)
            else:
                intercepts[key] = False
    return intercepts


h_int = 0
Test_int = []
for t_hail in itertools.combinations(range(len(hail_paths)), 2):
    intercepts = calc_intercepts(hail_paths[t_hail[0]], hail_paths[t_hail[1]])
    Test_int.append(intercepts)
    if intercepts['xy']:
        X = intercepts['xy'][0]
        Y = intercepts['xy'][1]
        if test_min <= X <= test_max and test_min <= Y <= test_max:
            h_int += 1

print('Part 1: ', h_int)


def calc_distance(hail1, hail2):
    r1 = np.array(hail1[:3])
    r2 = np.array(hail2[:3])
    e1 = np.array(hail1[3:])
    e2 = np.array(hail2[3:])
    n0 = np.cross(e1, e2)
    if n0.any():
        distance = np.abs(n0.dot(r1 - r2)) / np.linalg.norm(n0)
        return distance
    else:
        distance = np.linalg.norm(np.cross((r2 - r1), e1)) / (np.linalg.norm(e1))
        return distance


def get_all_intercepts(data_a):
    temp_int = []
    for t_hail in itertools.combinations(range(len(data_a)), 2):
        intercepts = calc_distance(data_a[t_hail[0]], data_a[t_hail[1]])
        temp_int.append(intercepts)
    return temp_int


def get_average_intercepts(data_i):
    output = {
        'average': sum(data_i) / len(data_i),
        'count': len(data_i)
    }
    return output


test_memo = {}


def test_offset(data_0, dims):
    if dims in test_memo:
        return test_memo[dims]
    x, y, z = dims
    temp_d = []
    for d in data_0:
        temp_d.append((d[0], d[1], d[2], d[3] + x, d[4] + y, d[5] + z))
    test_memo[dims] = get_average_intercepts(get_all_intercepts(temp_d))
    return test_memo[dims]


test_range = (-300, -250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 300)


def test_all(data_0, dims):
    dir_out = {}
    for x in test_range:
        for y in test_range:
            for z in test_range:
                test_dims = (dims[0] + x, dims[1] + y, dims[2] + z)
                dir_out[test_dims] = test_offset(data_0, test_dims)
    return dir_out


test_data = random.sample(hail_paths, 8)

start_test = (0, 0, 0)

best_offset = [start_test, test_offset(test_data, start_test)]
tested = {start_test: best_offset[1]}
best_found = False

count = 0
while not best_found:
    count += 1
    if count % 1 == 0:
        print(f'\rTest {count} best {best_offset}               ', end='', flush=True)
    test_offsets = test_all(test_data, best_offset[0])
    if count == 1:
        test_range = (-10, -6, -3, -1, 0, 1, 3, 6, 10)
    new_best = best_offset[0]
    for key, offset in test_offsets.items():
        if key not in tested:
            tested[key] = (offset, count)
        if offset['average'] < test_offsets[new_best]['average']:
            new_best = key
    if new_best != best_offset[0]:
        best_offset = [new_best, test_offsets[new_best]]
    else:
        best_found = True
        h1 = test_data[0]
        h2 = test_data[1]
        h1 = (h1[0], h1[1], h1[2], h1[3] + best_offset[0][0], h1[4] + best_offset[0][1], h1[5] + best_offset[0][2])
        h2 = (h2[0], h2[1], h2[2], h2[3] + best_offset[0][0], h2[4] + best_offset[0][1], h2[5] + best_offset[0][2])
        best_offset.append(calc_intercepts(h1, h2))
if best_offset[1]['average'] == 0:
    print('\nFound!')
else:
    print('\nNot quite found....')
print(best_offset)
total = best_offset[2]['xy'][0] + best_offset[2]['yz'][0] + best_offset[2]['xz'][1]
print('Part 2: ', int(total))
