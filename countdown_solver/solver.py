#!/usr/bin/env python3
"""
Countdown numbers‑game solver (exhaustive search, original 2018 core by
David Llewellyn‑Jones, MIT licence) wrapped as a library.
"""

import copy
import sys
from typing import List, Optional

_solution: Optional[str] = None
_max_op: int = 3

class _SolutionFound(Exception):
    pass


def findprev(layer, cols):
    found = -1
    pos = cols - 1
    while pos >= 0 and found < 0:
        if layer[pos] == 1:
            found = pos
        pos -= 1
    return found


def shuntup(layer, end):
    movedto = -1
    last = findprev(layer, end)
    if last >= 0:
        if last == end - 1:
            movedto = shuntup(layer, last)
            if movedto >= 0:
                movedto += 1
                layer[last] = 0
                layer[movedto] = 1
        else:
            layer[last] = 0
            layer[last + 1] = 1
            movedto = last + 1
    return movedto


def evaluate(formula, operators, numbers):
    calc = float("nan")
    left = formula[0]
    op = operators[formula[1]]
    right = formula[2]

    if isinstance(left, list):
        left = evaluate(left, operators, numbers)
    else:
        left = numbers[left]

    if isinstance(right, list):
        right = evaluate(right, operators, numbers)
    else:
        right = numbers[right]

    if op == 0:
        calc = left + right
    elif op == 1:
        calc = left - right
    elif op == 2:
        calc = left * right
    elif op == 3 and right != 0 and left % right == 0:
        calc = left / right
    elif op == 4:
        calc = left
    elif op == 5:
        calc = right
    return calc


def formulatostring(formula, operators, numbers):
    left = formula[0]
    op = operators[formula[1]]
    right = formula[2]

    if isinstance(left, list):
        left = formulatostring(left, operators, numbers)
    else:
        left = str(numbers[left])

    if isinstance(right, list):
        right = formulatostring(right, operators, numbers)
    else:
        right = str(numbers[right])

    if op == 0:
        return f"({left} + {right})"
    if op == 1:
        return f"({left} - {right})"
    if op == 2:
        return f"({left} * {right})"
    if op == 3:
        return f"({left} / {right})"
    if op == 4:
        return left
    if op == 5:
        return right


def permutations(elements):
    if len(elements) <= 1:
        yield elements
    else:
        for perm in permutations(elements[1:]):
            for i in range(len(elements)):
                yield perm[:i] + elements[0:1] + perm[i:]


def buildformula(built):
    formula = copy.deepcopy(built)
    leaf = 0
    operator = 0
    for layer in range(len(built)):
        if built[layer] != 0:
            for op in range(len(built[layer])):
                if built[layer][op] == 1:
                    formula[layer][op] = [0, operator, 0]
                    operator += 1
                else:
                    formula[layer][op] = leaf
                    leaf += 1
    for layer in range(len(built)):
        nextpos = 0
        if built[layer] != 0:
            for op in range(len(built[layer])):
                if built[layer][op] == 1:
                    if layer < len(built) - 1 and built[layer + 1] != 0:
                        formula[layer][op][0] = formula[layer + 1][nextpos]
                        formula[layer][op][2] = formula[layer + 1][nextpos + 1]
                        nextpos += 2
                    else:
                        formula[layer][op][0] = leaf
                        formula[layer][op][2] = leaf + 1
                        leaf += 2
                        nextpos += 2
    return formula[0]


def shuntoperator(operators, pos):
    if pos >= 0:
        operators[pos] += 1
        if operators[pos] > _max_op:
            operators[pos] = 0
            return shuntoperator(operators, pos - 1)
        return True
    return False


def chooselayer(connections, built, start, height, numbers, find, closest):
    global _solution
    if start < height and connections[start] > 0:
        hangers = 2 * connections[start - 1]
        coats = connections[start]
        if hangers >= coats:
            layer = [1] * coats + [0] * (hangers - coats)
            movedto = hangers
            while movedto >= 0:
                built[start] = layer
                closest = chooselayer(connections, built, start + 1, height, numbers, find, closest)
                movedto = shuntup(layer, hangers)
    else:
        formula = buildformula(built)
        for perm in permutations(numbers):
            operators = [0] * height
            more = True
            while more:
                more = shuntoperator(operators, height - 1)
                calc = evaluate(formula[0], operators, perm)
                if calc == calc:
                    distance = abs(calc - find)
                    if distance == 0:
                        _solution = formulatostring(formula[0], operators, perm)
                        raise _SolutionFound
                    if distance < closest:
                        closest = distance
    return closest


def shuntconnection(connections, height, maxnodes, nodes):
    changed = 0
    if height >= 1:
        current = connections[height - 1]
        nodesbelow = nodes - current
        prev = 0 if height <= 1 else connections[height - 2]
        if current >= 2 ** prev or nodesbelow + current >= maxnodes:
            changed = shuntconnection(connections, height - 1, maxnodes, nodesbelow)
            if nodesbelow + changed < maxnodes:
                connections[height - 1] = 1
                changed = changed - current + 1
                if height == maxnodes:
                    changed = -1
            else:
                connections[height - 1] = 0
                changed = changed - current
        else:
            connections[height - 1] = current + 1
            changed = 1
    return changed


def _connections(numbers: List[int], find: int):
    nodes = len(numbers) - 1
    conns = [1] * nodes
    changed = 0
    closest = 10 ** 10
    while changed == 0:
        built = [[1]] + [0] * (nodes - 1)
        closest = chooselayer(conns, built, 1, nodes, numbers, find, closest)
        changed = shuntconnection(conns, nodes, nodes, nodes)


def solve(numbers: List[int], target: int, require_all: bool = True) -> Optional[str]:
    global _solution, _max_op
    _solution = None
    _max_op = 3 if require_all else 5
    try:
        _connections(numbers, target)
    except _SolutionFound:
        pass
    return _solution


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("Usage: countdown_solver.py n1 n2 ... nk target")
    *nums, tgt = map(int, sys.argv[1:])
    res = solve(nums, tgt)
    print(res if res else "No solution")
