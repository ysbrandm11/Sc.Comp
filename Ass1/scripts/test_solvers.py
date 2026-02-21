"""
Important!!
AI GENERATED CODE
This code is not claimed as original work by the author of this repository.

This file was produced using OpenAI's ChatGPT.
It was generated in response to a prompt requesting test code for
Jacobi, Gauss-Seidel, and SOR iterative methods. The following prompt is used;

"" Hi chat, I have code that implements the Jacobi method,
the Gauss Seidel method and the SOR method,
can you write a few tests to see if they work? ""

The code has been reviewed, and slightly modified. However to repeat this code
is NOT claimed as own work!

This repository is for an educational assignment, these tests are not meant to
be used in the assignment neither do they help to solve the assignment,
they are only meant to be used as a sanity check for the implementations of the
iterative methods.
"""

# run_iterative_method_tests.py
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Dict

import numpy as np


# =========================
# CONFIG: EDIT THIS
# =========================
IMPORT_MODULE = "Solvers"  # <-- change to your filename/module (without .py)

# If your function signatures differ, edit the wrappers in load_solvers().
# Assumed signatures:
#   jacobi(A, b, x0=None, tol=1e-10, max_iter=5000) -> x or (x, iters) or (x, iters, hist) or dict
#   gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=5000) -> same
#   sor(A, b, omega, x0=None, tol=1e-10, max_iter=5000) -> same


# =========================
# Small test framework
# =========================
@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""


def _ok(name: str) -> TestResult:
    return TestResult(name=name, passed=True)


def _fail(name: str, msg: str) -> TestResult:
    return TestResult(name=name, passed=False, message=msg)


def assert_true(cond: bool, msg: str = "assertion failed") -> None:
    if not cond:
        raise AssertionError(msg)


def assert_allclose(a: np.ndarray, b: np.ndarray, rtol: float, atol: float, msg: str = "") -> None:
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.linalg.norm(a - b)
        raise AssertionError(msg or f"arrays not close (||a-b||={diff:g})")


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(A @ x - b, ord=2))


def spd_test_system(n: int = 6, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a system that should converge for Jacobi/GS/SOR.
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    A = M.T @ M  # SPD
    A += n * np.eye(n)  # make it nicer
    x_true = rng.normal(size=n)
    b = A @ x_true
    return A, b, x_true


def unpack_output(out: Any) -> Tuple[np.ndarray, Optional[int], Optional[np.ndarray]]:
    """
    Accept common patterns:
      - x
      - (x, iters)
      - (x, iters, history)
      - {"x":..., "iters":..., "history":...}
    """
    if isinstance(out, dict):
        x = out.get("x", out.get("solution"))
        iters = out.get("iters", out.get("n_iters", out.get("iterations")))
        hist = out.get("history", out.get("residuals", out.get("errors")))
        x = np.asarray(x, dtype=float)
        hist = None if hist is None else np.asarray(hist, dtype=float)
        return x, iters, hist

    if isinstance(out, tuple):
        if len(out) == 0:
            raise ValueError("solver returned empty tuple")
        x = np.asarray(out[0], dtype=float)
        iters = out[1] if len(out) >= 2 else None
        hist = np.asarray(out[2], dtype=float) if len(out) >= 3 and out[2] is not None else None
        return x, iters, hist

    x = np.asarray(out, dtype=float)
    return x, None, None


def load_solvers() -> Tuple[
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
]:
    mod = importlib.import_module(IMPORT_MODULE)

    def jacobi(A, b, x0=None, tol=1e-10, max_iter=5000):
        return mod.Jacobi(A, b, eps=tol, max_iterations=max_iter)

    def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=5000):
        return mod.gauss_seidel(A, b, eps=tol, max_iterations=max_iter)

    def sor(A, b, omega, x0=None, tol=1e-10, max_iter=5000):
        return mod.sor(A, b, omega=omega, eps=tol, max_iterations=max_iter)

    return jacobi, gauss_seidel, sor


# =========================
# Tests
# =========================
def test_solver_matches_direct(name: str, solve_fn: Callable[..., Any], A: np.ndarray, b: np.ndarray, **kwargs) -> None:
    out = solve_fn(A, b, **kwargs)
    x, iters, hist = unpack_output(out)

    n = A.shape[0]
    assert_true(x.shape == (n,), f"{name}: expected shape ({n},), got {x.shape}")

    r = residual_norm(A, x, b)
    assert_true(r < 1e-7, f"{name}: residual too large: {r:g}")

    x_star = np.linalg.solve(A, b)
    assert_allclose(x, x_star, rtol=1e-6, atol=1e-8, msg=f"{name}: solution differs from np.linalg.solve")

    if iters is not None:
        assert_true(int(iters) > 0, f"{name}: iterations should be > 0")

    if hist is not None:
        assert_true(hist.ndim == 1, f"{name}: history should be 1D")
        assert_true(np.isfinite(hist).all(), f"{name}: history has NaN/inf")
        assert_true(hist[-1] < 1e-7, f"{name}: last history value not small: {hist[-1]:g}")
        assert_true(float(hist.max()) < 1e6, f"{name}: history exploded: max={hist.max():g}")


def test_sor_omega_1_equals_gs(gauss_seidel, sor) -> None:
    A, b, _ = spd_test_system(n=6, seed=999)
    x0 = np.zeros(6)

    x_gs, _, _ = unpack_output(gauss_seidel(A, b, x0=x0, tol=1e-12, max_iter=50_000))
    x_sor, _, _ = unpack_output(sor(A, b, omega=1.0, x0=x0, tol=1e-12, max_iter=50_000))

    assert_allclose(x_sor, x_gs, rtol=1e-10, atol=1e-12, msg="SOR with omega=1 should match Gauss-Seidel")


def test_likely_divergent_does_not_fake_success(name: str, solve_fn: Callable[..., Any]) -> None:
    A = np.array([[1.0, 3.0],
                  [2.0, 1.0]])
    b = np.array([1.0, 1.0])
    x0 = np.zeros(2)

    tol = 1e-12
    max_iter = 50

    try:
        out = solve_fn(A, b, x0=x0, tol=tol, max_iter=max_iter)
    except Exception:
        # Good: solver explicitly signals failure
        return

    x, iters, hist = unpack_output(out)
    r = residual_norm(A, x, b)

    # If it "converged" here, either your implementation is unusually robust
    # or the test matrix isn't divergent for your update rule.
    assert_true(r > 1e-6, f"{name}: unexpectedly reached tiny residual on a likely divergent system (r={r:g})")


def test_sor_invalid_omega_rejected(sor) -> None:
    A, b, _ = spd_test_system(n=5, seed=7)
    x0 = np.zeros(5)

    for bad_omega in (-0.1, 0.0, 2.0, 2.5):
        ok = False
        try:
            sor(A, b, omega=bad_omega, x0=x0, tol=1e-10, max_iter=1000)
        except Exception:
            ok = True
        assert_true(ok, f"SOR should reject omega={bad_omega} (or you can remove this test if you allow it)")


# =========================
# Runner
# =========================
def run() -> int:
    results: list[TestResult] = []

    try:
        jacobi, gauss_seidel, sor = load_solvers()
    except Exception as e:
        print(f"ERROR: Could not import/load solvers from module '{IMPORT_MODULE}': {e}")
        return 2

    # Core accuracy tests on SPD systems
    for n in (3, 6, 10):
        A, b, _ = spd_test_system(n=n, seed=123 + n)
        x0 = np.zeros(n)

        for name, fn, kwargs in [
            ("jacobi", jacobi, dict(x0=x0, tol=1e-10, max_iter=50_000)),
            ("gauss_seidel", gauss_seidel, dict(x0=x0, tol=1e-10, max_iter=50_000)),
        ]:
            test_name = f"{name}_matches_direct_n={n}"
            try:
                test_solver_matches_direct(name, fn, A, b, **kwargs)
                results.append(_ok(test_name))
            except Exception as e:
                results.append(_fail(test_name, str(e)))

        # SOR for a few omegas
        for omega in (0.5, 1.0, 1.25, 1.8):
            test_name = f"sor_matches_direct_n={n}_omega={omega}"
            try:
                test_solver_matches_direct("sor", lambda AA, bb, **kw: sor(AA, bb, omega=omega, **kw),
                                          A, b, x0=x0, tol=1e-10, max_iter=50_000)
                results.append(_ok(test_name))
            except Exception as e:
                results.append(_fail(test_name, str(e)))

    # Property test: omega=1 equals GS
    try:
        test_sor_omega_1_equals_gs(gauss_seidel, sor)
        results.append(_ok("sor_omega_1_equals_gauss_seidel"))
    except Exception as e:
        results.append(_fail("sor_omega_1_equals_gauss_seidel", str(e)))

    # Likely-divergent case: should not pretend it converged
    for name, fn in [
        ("jacobi", jacobi),
        ("gauss_seidel", gauss_seidel),
        ("sor(omega=1.1)", lambda A, b, **kw: sor(A, b, omega=1.1, **kw)),
    ]:
        test_name = f"{name}_likely_divergent_not_fake_success"
        try:
            test_likely_divergent_does_not_fake_success(name, fn)
            results.append(_ok(test_name))
        except Exception as e:
            results.append(_fail(test_name, str(e)))

    # Optional: invalid omega rejected
    try:
        test_sor_invalid_omega_rejected(sor)
        results.append(_ok("sor_invalid_omega_rejected"))
    except Exception as e:
        results.append(_fail("sor_invalid_omega_rejected", str(e)))

    # Print summary
    width = max(len(r.name) for r in results) if results else 10
    passed = sum(r.passed for r in results)
    total = len(results)

    print("\n=== Iterative Solver Tests (no pytest) ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{status:4}  {r.name:<{width}}  {'' if r.passed else r.message}")

    print(f"\nSummary: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(run())