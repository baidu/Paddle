# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from .utils import _value_and_gradient


def cubic_interpolation_(x1, f1, g1, x2, f2, g2):
    r"""Cubic interpolation between (x1, f1, g1) and (x2, f2, g2).
        Use two points and their gradient to determine a cubic function and get the minimum point
        between them in the cubic curve.

    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006.
        pp59: formula 3.59

    Args:
        x1, f1, g1: point1's position, value and gradient.
        x2, f2, g2: point2's position, value and gradient.
    Returns:
        min_pos: the minimum point between the specified points in the cubic curve.
    """
    xmin, xmax = paddle.static.nn.cond(
        x1 <= x2, lambda: (x1, x2), lambda: (x2, x1)
    )
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2

    def true_func1():
        d2 = d2_square.sqrt()

        def true_fn2():
            return x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))

        def false_fn2():
            return x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))

        pred = paddle.less_equal(x=x1, y=x2)
        min_pos = paddle.static.nn.cond(pred, true_fn2, false_fn2)

        return paddle.minimum(paddle.maximum(min_pos, xmin), xmax)

    def false_func1():
        return (xmin + xmax) / 2.0

    min_pos = paddle.static.nn.cond(d2_square >= 0.0, true_func1, false_func1)
    return min_pos


def strong_wolfe(
    f,
    xk,
    pk,
    max_iters=20,
    tolerance_change=1e-8,
    initial_step_length=1.0,
    c1=1e-4,
    c2=0.9,
    alpha_max=10,
    dtype='float32',
):
    r"""Implements of line search algorithm that satisfies the strong Wolfe conditions using double zoom.

    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006.
        pp60: Algorithm 3.5 (Line Search Algorithm).

    Args:
        f: the objective function to minimize. ``f`` accepts a multivariate input and returns a scalar.
        xk (Tensor): the starting point of the iterates.
        pk (Tensor): search direction.
        max_iters (Scalar): the maximum number of iterations.
        tolerance_grad (Scalar): terminates if the gradient norm is smaller than
            this. Currently gradient norm uses inf norm.
        tolerance_change (Scalar): terminates if the change of function value/position/parameter between
            two iterations is smaller than this value.
        initial_step_length (Scalar): step length used in first iteration.
        c1 (Scalar): parameter for sufficient decrease condition.
        c2 (Scalar): parameter for curvature condition.
        alpha_max (float): max step length.
        dtype ('float32' | 'float64'): the datatype to be used.

    Returns:
        num_func_calls (float): number of objective function called in line search process.
        a_star(Tensor): optimal step length, or 0. if the line search algorithm did not converge.
        phi_star (Tensor): phi at a_star.
        derphi_star (Tensor): derivative of phi at a_star.

    Following summarizes the essentials of the strong Wolfe line search algorithm.
    Some notations used in the description:

        - `f` denotes the objective function.
        - `phi` is a function of step size alpha, restricting `f` on a line.

            phi = f(xk + a * pk),
            where xk is the position of k'th iterate, pk is the line search direction(decent direction),
            and a is the step size.
        - a : substitute of alpha
        - a1 is a of last iteration, which is alpha_(i-1).
        - a2 is a of current iteration, which is alpha_i.
        - a_lo is a in left position when calls zoom, which is alpha_low.
        - a_hi is a in right position when calls zoom, which is alpha_high.

    Line Search Algorithm:
        repeat
            Compute phi(a2) and derphi(a2).
            1. If phi(a2) > phi(0) + c_1 * a2 * phi'(0) or [phi(a2) >= phi(a1) and i > 1],
                a_star= zoom(a1, a2) and stop;

            2. If |phi'(a2)| <= -c_2 * phi'(0),
                a_star= a2 and stop;

            3. If phi'(a2) >= 0,
                a_star= zoom(a2, a1) and stop;

            a1 = a2
            a2 = min(2 * a2, a2)
            i = i + 1
        end(repeat)

    zoom(a_lo, a_hi) Algorithm:
        repeat
            aj = cubic_interpolation(a_lo, a_hi)
            Compute phi(aj) and derphi(aj).
            1. If phi(aj) > phi(0) + c_1 * aj * phi'(0) or phi(aj) >= phi(a_lo),
                then a_hi <- aj;
            2.
                2.1. If |phi'(aj)| <= -c_2 * phi'(0), then a_star= a2 and stop;

                2.2. If phi'(aj) * (a2 - a1) >= 0, then a_hi = a_lo

                a_lo = aj;
        end(repeat)
    """

    def phi_and_derphi(a):
        r"""Compute function value and derivative of phi at a.
        phi = f(xk + a * pk)
        phi'(a) = f'(xk + a * pk) * pk
        """
        phi_value, f_grad = _value_and_gradient(f, xk + a * pk)
        phi_grad = paddle.dot(f_grad, pk)
        # return f_grad to be used in bfgs/l-bfgs to compute yk to avoid computint repeatedly.
        return phi_value, f_grad, phi_grad

    def zoom(
        a_lo,
        phi_lo,
        derphi_lo,
        derf_lo,
        a_hi,
        phi_hi,
        derphi_hi,
        phi_0,
        derphi_0,
    ):
        # find the exact a from the bracket [a_lo, a_hi]
        max_zoom_iters = max_iters
        j = paddle.full(shape=[1], fill_value=0, dtype='int64')
        done_zoom = paddle.full(shape=[1], fill_value=False, dtype='bool')

        def cond_zoom(
            j,
            done_zoom,
            a_lo,
            phi_lo,
            derphi_lo,
            derf_lo,
            a_hi,
            phi_hi,
            derphi_hi,
        ):
            pred = paddle.abs(a_hi - a_lo) < tolerance_change
            paddle.assign(done_zoom | pred, done_zoom)
            return (j < max_zoom_iters) & ~done_zoom

        def body_zoom(
            j,
            done_zoom,
            a_lo,
            phi_lo,
            derphi_lo,
            derf_lo,
            a_hi,
            phi_hi,
            derphi_hi,
        ):
            aj = cubic_interpolation_(
                a_lo, phi_lo, derphi_lo, a_hi, phi_hi, derphi_hi
            )  # 21
            min_change = 0.1 * paddle.abs(a_hi - a_lo)
            pred = (
                paddle.minimum(paddle.abs(aj - a_lo), paddle.abs(aj - a_hi))
                < min_change
            )
            aj = paddle.static.nn.cond(
                pred, lambda: 0.5 * (a_lo + a_hi), lambda: aj
            )

            phi_j, derf_j, derphi_j = phi_and_derphi(aj)

            def true_fn():
                # use assign to modify the variable in-place
                paddle.assign(aj, a_hi)
                paddle.assign(phi_j, phi_hi)
                paddle.assign(derphi_j, derphi_hi)

            def false_fn(a_lo, done_zoom):
                pred3 = paddle.abs(derphi_j) <= -c2 * derphi_0
                paddle.assign(pred3, done_zoom)

                def true_fn():
                    paddle.assign(a_lo, a_hi)
                    paddle.assign(phi_lo, phi_hi)
                    paddle.assign(derphi_lo, derphi_hi)

                pred4 = ~done_zoom & (derphi_j * (a_hi - a_lo) >= 0)
                paddle.static.nn.cond(pred4, true_fn, None)

                paddle.assign(aj, a_lo)
                paddle.assign(phi_j, phi_lo)
                paddle.assign(derphi_j, derphi_lo)
                paddle.assign(derf_j, derf_lo)

            pred2 = (phi_j > phi_0 + c1 * aj * derphi_0) | (phi_j >= phi_lo)
            paddle.static.nn.cond(
                pred2, true_fn, lambda: false_fn(a_lo, done_zoom)
            )
            j = paddle.static.nn.cond(done_zoom, lambda: j, lambda: j + 1)
            return [
                j,
                done_zoom,
                a_lo,
                phi_lo,
                derphi_lo,
                derf_lo,
                a_hi,
                phi_hi,
                derphi_hi,
            ]

        paddle.static.nn.while_loop(
            cond=cond_zoom,
            body=body_zoom,
            loop_vars=[
                j,
                done_zoom,
                a_lo,
                phi_lo,
                derphi_lo,
                derf_lo,
                a_hi,
                phi_hi,
                derphi_hi,
            ],
        )
        # j is the number of object function called in zoom.
        return j

    alpha_max = paddle.full(shape=[1], fill_value=alpha_max, dtype=dtype)

    a1 = paddle.full(shape=[1], fill_value=0.0, dtype=dtype)
    a2 = paddle.full(shape=[1], fill_value=initial_step_length, dtype=dtype)

    phi_1, derf_1, derphi_1 = phi_and_derphi(a1)
    # use assign to cut off binding between two variables
    phi_0 = paddle.assign(phi_1)
    derphi_0 = paddle.assign(derphi_1)
    ls_func_calls = paddle.full(shape=[1], fill_value=1, dtype='int64')

    # If not found the a_star, will return alpha=0 and f(xk), derf(xk)
    a_star = paddle.full(shape=[1], fill_value=0, dtype=dtype)
    phi_star = paddle.assign(phi_1)
    derf_star = paddle.assign(derf_1)

    i = paddle.full(shape=[1], fill_value=0, dtype='int64')
    done = paddle.full(shape=[1], fill_value=False, dtype='bool')

    def cond(i, ls_func_calls, a1, a2, phi_1, derf_1, done):
        return (i < max_iters) & ~done

    def body(i, ls_func_calls, a1, a2, phi_1, derf_1, done):
        phi_2, derf_2, derphi_2 = phi_and_derphi(a2)
        paddle.assign(ls_func_calls + 1, ls_func_calls)
        paddle.assign(done | paddle.any(paddle.isinf(phi_2)), done)

        def true_fn1():
            j = zoom(
                a1,
                phi_1,
                derphi_1,
                derf_1,
                a2,
                phi_2,
                derphi_2,
                phi_0,
                derphi_0,
            )
            paddle.assign(a1, a_star)
            paddle.assign(phi_1, phi_star)
            paddle.assign(derf_1, derf_star)
            paddle.assign(ls_func_calls + j, ls_func_calls)

        pred1 = ~done & (
            (phi_2 > phi_0 + c1 * a2 * derphi_0) | ((phi_2 >= phi_1) & (i > 1))
        )
        paddle.assign(done | pred1, done)
        paddle.static.nn.cond(pred1, true_fn1, None)

        def true_fn2():
            paddle.assign(a2, a_star)
            paddle.assign(phi_2, phi_star)
            paddle.assign(derf_2, derf_star)

        pred2 = ~done & (paddle.abs(derphi_2) <= -c2 * derphi_0)
        paddle.assign(done | pred2, done)
        paddle.static.nn.cond(pred2, true_fn2, None)

        def true_fn3():
            j = zoom(
                a2,
                phi_2,
                derphi_2,
                derf_2,
                a1,
                phi_1,
                derphi_1,
                phi_0,
                derphi_0,
            )
            paddle.assign(a2, a_star)
            paddle.assign(phi_2, phi_star)
            paddle.assign(derf_2, derf_star)
            paddle.assign(ls_func_calls + j, ls_func_calls)

        pred3 = ~done & (derphi_2 >= 0)
        paddle.assign(done | pred3, done)
        paddle.static.nn.cond(pred3, true_fn3, None)

        def false_fn():
            paddle.assign(a2, a1)
            paddle.assign(phi_2, phi_1)
            paddle.assign(derf_2, derf_1)
            paddle.assign(paddle.minimum(2 * a2, alpha_max), a2)
            paddle.assign(i + 1, i)

        paddle.static.nn.cond(done, None, false_fn)
        return [i, ls_func_calls, a1, a2, phi_1, derf_1, done]

    paddle.static.nn.while_loop(
        cond=cond,
        body=body,
        loop_vars=[i, ls_func_calls, a1, a2, phi_1, derf_1, done],
    )

    return a_star, phi_star, derf_star, ls_func_calls
