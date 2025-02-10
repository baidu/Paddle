# Prim

Test composite and primitive API and related autodiff rules.


## Package Structure

- comp(abbr for composite):

  Test composite API which is composed of primitive api, but not include the autodiff rules for primitive api.

- prim(abbr for primitive):

  Test primitive api and autodiff rules.

    - vjp: Test vjp rules.

        - eager: Test vjp rules in eager mode.
        - static: Test vjp rules in static mode.

    - jvp(TODO): Test jvp rules.


## How to

- Forward API and First-order Autodiff

  Compare numerical value with raw phi operators using tools such as `np.testing.assert_allclose`


- Higher-order Autodiff

  Compare numerical value with the result computed by finite difference. Tool
  used for computing Higher-order finite difference will be provided in the future.
