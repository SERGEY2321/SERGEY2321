"""
Convert a raw OCR string into a SymPy-compatible expression.
"""

import re
import sympy as sp


# Common OCR mistakes and shorthands to fix before parsing
_FIXES = [
    # spaces around operators
    (r"\s*\*\*\s*", "**"),
    # implicit multiplication: 2x → 2*x, 3(x+1) → 3*(x+1)
    (r"(\d)([a-zA-Z(])", r"\1*\2"),
    (r"([a-zA-Z)])(\d)", r"\1*\2"),
    (r"([a-zA-Z])\(", r"\1*("),
    # sqrt shorthand
    (r"sqrt\s*\(", "sqrt("),
    # remove stray spaces inside expressions
    (r"\s*([\+\-\*\/\^=\(\)])\s*", r" \1 "),
]


def _fix_expression(expr: str) -> str:
    for pattern, replacement in _FIXES:
        expr = re.sub(pattern, replacement, expr)
    return expr.strip()


def parse(raw: str) -> dict:
    """
    Parse a raw expression string.

    Returns a dict:
      {
        "type": "equation" | "expression",
        "lhs": sympy expr,     # left-hand side  (for equations)
        "rhs": sympy expr,     # right-hand side (for equations)
        "expr": sympy expr,    # full expression (for pure expressions)
        "symbols": list[sympy.Symbol],
        "raw_fixed": str,      # cleaned string fed to SymPy
      }

    Raises ValueError if SymPy cannot parse the input.
    """
    fixed = _fix_expression(raw)

    if "=" in fixed:
        # Equation: split into LHS and RHS
        parts = fixed.split("=", 1)
        lhs_str = parts[0].strip()
        rhs_str = parts[1].strip()
        try:
            lhs = sp.sympify(lhs_str, evaluate=False)
            rhs = sp.sympify(rhs_str, evaluate=False)
        except Exception as e:
            raise ValueError(f"Cannot parse equation '{fixed}': {e}")
        syms = sorted(lhs.free_symbols | rhs.free_symbols, key=str)
        return {
            "type": "equation",
            "lhs": lhs,
            "rhs": rhs,
            "expr": None,
            "symbols": syms,
            "raw_fixed": fixed,
        }
    else:
        # Pure expression (e.g. "2 + 3", "integrate(x**2, x)")
        try:
            expr = sp.sympify(fixed, evaluate=False)
        except Exception as e:
            raise ValueError(f"Cannot parse expression '{fixed}': {e}")
        syms = sorted(expr.free_symbols, key=str)
        return {
            "type": "expression",
            "lhs": None,
            "rhs": None,
            "expr": expr,
            "symbols": syms,
            "raw_fixed": fixed,
        }
