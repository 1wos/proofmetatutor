"""Deterministic arithmetic checker (sympy).

The TPU-trained mBERT verifier scores step *plausibility*, but does not compute,
so it cannot tell that x = 4 is wrong for 2x + 6 = 10. This rule-based checker
fills that gap honestly: it parses the equations in a step and verifies them
with a CAS. Combined, the demo gets a real "caught the mistake at step N" moment
without faking the model.

It is intentionally conservative: when it cannot parse a step it returns
verdict=None ("unknown") rather than guessing.
"""

from __future__ import annotations

import re
from typing import Any

from sympy import Eq, Rational, simplify, symbols
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_X = symbols("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_EQ_RE = re.compile(r"([0-9xX²\(\)\.\s\+\-\*/]+)=([0-9xX²\(\)\.\s\+\-\*/]+)")


def _norm(s: str) -> str:
    return s.replace("²", "**2").replace("×", "*").replace("÷", "/").replace("X", "x")


def _expr(s: str):
    return parse_expr(_norm(s).strip(), transformations=_TRANSFORMS, evaluate=True)


def _equations(text: str) -> list[tuple[Any, Any]]:
    out = []
    for lhs, rhs in _EQ_RE.findall(text):
        if not lhs.strip() or not rhs.strip():
            continue
        try:
            out.append((_expr(lhs), _expr(rhs)))
        except Exception:
            continue
    return out


def solve_problem(problem_text: str) -> set | None:
    """Return the set of correct answers for the problem, or None if unparseable."""
    try:
        eqs = _equations(problem_text)
        if eqs:
            from sympy import solve

            lhs, rhs = eqs[0]
            sols = solve(Eq(lhs, rhs), _X)
            return {simplify(s) for s in sols} if sols else None
        # No equation: treat as an expression to evaluate, e.g. "3/4 + 1/4".
        m = re.search(r"[0-9\(\)\.\s\+\-\*/]+", problem_text)
        if m and any(c.isdigit() for c in m.group()):
            return {simplify(_expr(m.group()))}
    except Exception:
        return None
    return None


def check_step(step_text: str, expected: set | None) -> dict[str, Any]:
    """Return {verdict: 'ok'|'error'|None, reason}."""
    eqs = _equations(step_text)
    if not eqs:
        return {"verdict": None, "reason": "수식을 찾지 못함"}

    try:
        # 1) Internal consistency: a fully numeric equation must actually hold.
        for lhs, rhs in eqs:
            if not lhs.free_symbols and not rhs.free_symbols:
                if simplify(lhs - rhs) != 0:
                    return {"verdict": "error", "reason": f"{lhs} ≠ {rhs} (계산 오류)"}

        # 2) Answer check: if the step claims a value, compare to the solution.
        if expected:
            for lhs, rhs in eqs:
                # "x = N"
                if lhs == _X and not rhs.free_symbols:
                    if simplify(rhs) not in expected:
                        exp = ", ".join(str(e) for e in expected)
                        return {"verdict": "error", "reason": f"x = {rhs} (정답은 {exp})"}
                # standalone numeric result as the step's answer
            last_lhs, last_rhs = eqs[-1]
            if not last_rhs.free_symbols and not last_lhs.free_symbols:
                val = simplify(last_rhs)
                if val not in expected and simplify(last_lhs) not in expected:
                    exp = ", ".join(str(e) for e in expected)
                    return {"verdict": "error", "reason": f"답 {val} ≠ 정답 {exp}"}
    except Exception:
        return {"verdict": None, "reason": "검증 불가"}

    return {"verdict": "ok", "reason": "계산 일치"}


def check_solution(problem_text: str, steps: list[str]) -> dict[str, Any]:
    expected = solve_problem(problem_text)
    results = []
    first_error = -1
    for i, s in enumerate(steps):
        r = check_step(s, expected)
        results.append({"step": s, **r})
        if first_error < 0 and r["verdict"] == "error":
            first_error = i
    return {
        "expected": [str(e) for e in expected] if expected else None,
        "steps": results,
        "first_error_index": first_error,
    }
