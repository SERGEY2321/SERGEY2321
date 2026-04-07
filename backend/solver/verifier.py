"""
Verification helpers — substitute answers back and confirm correctness.
"""

import sympy as sp


def verify_equation(
    equation: sp.Eq,
    variable: sp.Symbol,
    solutions: list,
) -> tuple[bool, list[str]]:
    """
    Substitute each solution back into the equation and check equality.

    Returns:
        (all_verified: bool, steps: list[str])
    """
    steps: list[str] = []
    steps.append("── Проверка ──")
    all_ok = True

    for sol in solutions:
        lhs_val = sp.simplify(equation.lhs.subs(variable, sol))
        rhs_val = sp.simplify(equation.rhs.subs(variable, sol))
        diff = sp.simplify(lhs_val - rhs_val)
        ok = diff == 0
        if not ok:
            all_ok = False
        mark = "✓" if ok else "✗"
        steps.append(
            f"  {variable} = {sol}:  LHS = {lhs_val},  RHS = {rhs_val}  {mark}"
        )

    steps.append(f"Проверка {'пройдена' if all_ok else 'не пройдена'}")
    return all_ok, steps


def verify_expression(
    original: sp.Expr,
    simplified: sp.Expr,
) -> tuple[bool, list[str]]:
    """
    Verify that simplified expression equals the original.

    Returns:
        (verified: bool, steps: list[str])
    """
    steps: list[str] = []
    steps.append("── Проверка ──")
    diff = sp.simplify(original - simplified)
    ok = diff == 0
    mark = "✓" if ok else "✗"
    steps.append(f"  original − simplified = {diff}  {mark}")
    steps.append(f"Проверка {'пройдена' if ok else 'не пройдена'}")
    return ok, steps
