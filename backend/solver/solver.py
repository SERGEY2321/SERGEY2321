"""
Math solver — uses SymPy to solve equations and evaluate expressions.
Returns a structured result with an answer and step-by-step explanation.
"""

from __future__ import annotations
import sympy as sp
from .parser import parse


def solve(raw_expression: str) -> dict:
    """
    Main entry point.

    Args:
        raw_expression: string from OCR, e.g. "2*x + 3 = 7"

    Returns:
        {
          "expression": str,       # cleaned expression shown to user
          "answer": str,           # human-readable answer
          "steps": list[str],      # step-by-step solution
          "verified": bool,        # whether verification passed
          "error": str | None,     # error message if solving failed
        }
    """
    try:
        parsed = parse(raw_expression)
    except ValueError as e:
        return _error(raw_expression, str(e))

    try:
        if parsed["type"] == "equation":
            return _solve_equation(parsed)
        else:
            return _evaluate_expression(parsed)
    except Exception as e:
        return _error(parsed["raw_fixed"], f"Solver error: {e}")


# ─── Equation solving ────────────────────────────────────────────────────────

def _solve_equation(parsed: dict) -> dict:
    lhs = parsed["lhs"]
    rhs = parsed["rhs"]
    syms = parsed["symbols"]
    expr_str = parsed["raw_fixed"]
    steps: list[str] = []

    steps.append(f"Вхідний вираз: {lhs} = {rhs}")

    equation = sp.Eq(lhs, rhs)
    steps.append(f"SymPy рівняння: {equation}")

    if not syms:
        # No unknowns — just check equality
        diff = sp.simplify(lhs - rhs)
        is_true = diff == 0
        answer = "Вірно" if is_true else "Невірно"
        steps.append(f"Різниця LHS − RHS = {diff}")
        steps.append(f"Відповідь: {answer}")
        return _result(expr_str, answer, steps, verified=is_true)

    # Differentiate: integral / differential / plain equation
    if _is_calculus(lhs) or _is_calculus(rhs):
        return _solve_calculus(parsed, steps)

    # Try to solve for each unknown
    variable = syms[0]
    steps.append(f"Розв'язуємо відносно: {variable}")

    solutions = sp.solve(equation, variable)
    if not solutions:
        steps.append("SymPy не знайшов розв'язків.")
        return _result(expr_str, "Немає розв'язків", steps, verified=False)

    if len(solutions) == 1:
        sol = solutions[0]
        sol_simplified = sp.simplify(sol)
        steps.append(f"Розв'язок: {variable} = {sol_simplified}")
        answer = f"{variable} = {sol_simplified}"
    else:
        simplified = [sp.simplify(s) for s in solutions]
        answer_parts = [f"{variable} = {s}" for s in simplified]
        steps.append("Знайдено кілька розв'язків:")
        for a in answer_parts:
            steps.append(f"  {a}")
        answer = ",  ".join(answer_parts)

    from .verifier import verify_equation
    verified, v_steps = verify_equation(equation, variable, solutions)
    steps.extend(v_steps)

    return _result(expr_str, answer, steps, verified)


# ─── Expression evaluation ───────────────────────────────────────────────────

def _evaluate_expression(parsed: dict) -> dict:
    expr = parsed["expr"]
    syms = parsed["symbols"]
    expr_str = parsed["raw_fixed"]
    steps: list[str] = []

    steps.append(f"Вираз: {expr}")

    if _is_integral(expr):
        return _compute_integral(expr, expr_str, steps)
    if _is_derivative(expr):
        return _compute_derivative(expr, expr_str, steps)

    simplified = sp.simplify(expr)
    steps.append(f"Спрощення: {simplified}")

    evaluated = sp.nsimplify(simplified)
    if evaluated != simplified:
        steps.append(f"nsimplify: {evaluated}")

    # Try numeric evaluation
    try:
        numeric = float(sp.N(simplified))
        if simplified != numeric:
            steps.append(f"Числове значення: {numeric}")
        answer = str(simplified) if syms else _format_number(numeric)
    except Exception:
        answer = str(simplified)

    from .verifier import verify_expression
    verified, v_steps = verify_expression(expr, simplified)
    steps.extend(v_steps)

    return _result(expr_str, answer, steps, verified)


# ─── Calculus helpers ────────────────────────────────────────────────────────

def _is_calculus(expr) -> bool:
    return isinstance(expr, (sp.Integral, sp.Derivative))


def _is_integral(expr) -> bool:
    return isinstance(expr, sp.Integral)


def _is_derivative(expr) -> bool:
    return isinstance(expr, sp.Derivative)


def _compute_integral(expr: sp.Integral, expr_str: str, steps: list) -> dict:
    steps.append(f"Інтеграл: {expr}")
    result = sp.integrate(*expr.args)
    result_simplified = sp.simplify(result)
    steps.append(f"Первісна: {result_simplified} + C")
    answer = f"{result_simplified} + C"
    return _result(expr_str, answer, steps, verified=True)


def _compute_derivative(expr: sp.Derivative, expr_str: str, steps: list) -> dict:
    steps.append(f"Похідна: {expr}")
    result = sp.diff(*expr.args)
    result_simplified = sp.simplify(result)
    steps.append(f"Похідна = {result_simplified}")
    answer = str(result_simplified)
    return _result(expr_str, answer, steps, verified=True)


def _solve_calculus(parsed: dict, steps: list) -> dict:
    """Fallback for equations containing integrals/derivatives."""
    lhs, rhs = parsed["lhs"], parsed["rhs"]
    expr_str = parsed["raw_fixed"]
    combined = sp.Eq(lhs, rhs)
    try:
        result = sp.dsolve(combined)
        steps.append(f"dsolve: {result}")
        answer = str(result)
    except Exception as e:
        steps.append(f"dsolve failed: {e}")
        answer = "Не вдалося розв'язати"
    return _result(expr_str, answer, steps, verified=False)


# ─── Utilities ───────────────────────────────────────────────────────────────

def _format_number(n: float) -> str:
    if n == int(n):
        return str(int(n))
    return f"{n:.6g}"


def _result(expression: str, answer: str, steps: list[str], verified: bool) -> dict:
    return {
        "expression": expression,
        "answer": answer,
        "steps": steps,
        "verified": verified,
        "error": None,
    }


def _error(expression: str, message: str) -> dict:
    return {
        "expression": expression,
        "answer": "",
        "steps": [],
        "verified": False,
        "error": message,
    }
