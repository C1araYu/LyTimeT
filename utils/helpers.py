import sympy
import numpy as np
from sympy.parsing.sympy_parser import parse_expr

def rewrite(eq):
    idx2str = {0: ')', 1: ') ** 2', 2: ') ** 3'}
    stack, new_eq = [], ''
    while len(eq) > 0:
        if eq.startswith('square'):
            stack.append(1)
            new_eq += '('
            eq = eq[7:]
        elif eq.startswith('cube'):
            stack.append(2)
            new_eq += '('
            eq = eq[5:]
        elif eq[0] == '(':
            stack.append(0)
            new_eq += '('
            eq = eq[1:]
        elif eq[0] == ')':
            pop = stack[-1]
            stack = stack[:-1]
            new_eq += idx2str[pop]
            eq = eq[1:]
        else:
            new_eq += eq[0]
            eq = eq[1:]
    return new_eq

def simplify(eq, const_threshold, full_simplify=True):
    final_eq = rewrite(eq)
    expr = parse_expr(final_eq)
    if full_simplify:
        # fully simplifying the expression may have time/memory issues
        eq, final_eq = None, sympy.simplify(expr)
    else:
        eq, final_eq = None, expr
    while not (eq == final_eq):
        eq = final_eq
        for const in sympy.preorder_traversal(eq):
            if isinstance(const, sympy.Float):
                const = float(const)
                if np.abs(const) < const_threshold:
                    final_eq = final_eq.subs(const, 0.0)
                elif const < 1.0:
                    rounded_const = float('%.2g' % const)
                    if not (rounded_const == const):
                        final_eq = final_eq.subs(const, float('%.2g' % const))
                else:
                    rounded_const = round(const, 2)
                    if not (rounded_const == const):
                        final_eq = final_eq.subs(const, round(const, 2))
    return str(final_eq)

def natural_sort(ids):
    video_frame = [id.replace('.png', '').split('_') for id in ids]
    keys = [int(vf[0])*10000 + int(vf[1]) for vf in video_frame]
    key_ids = [(k,i) for k,i in zip(keys, ids)]
    key_ids.sort()
    return [i for k,i in key_ids]