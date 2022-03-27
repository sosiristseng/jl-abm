# Agent-based Modeling

Agent-based Modeling (ABM) is a simulation method where the autonomous agents interacting with the environment (space) and/or each other by a set of rules.

The most obvious example of ABM is to simulate actions of non-player characters (NPCs) in computer games.

ABM is able to model heterogeneously, i.e. it does not require the environment to be well stirred (as opposed to ODEs), continuous (as opposed to to PDEs), nor need the characteristics of each kind of agents to be identical (as opposed to SSAs).

This makes ABM more flexible to model individual behaviors. (e.g. traffic jam, disease spread, molecular interactions)

## Elements of ABM

To use `Agents.jl`, we need to define:

- The [**space**](https://juliadynamics.github.io/Agents.jl/stable/api/#Available-spaces) where the agents live
- The [**agents**](https://juliadynamics.github.io/Agents.jl/stable/api/#@agent-macro) with self-defined properties.
- The **model** to hold the `space`, the `agent`s, and other parameters (called `properties`)
- The stepping function `step!()` to tell how the model evolve.

## Could I do ABM from scratch?


## Resources

- [Documentation](https://juliadynamics.github.io/Agents.jl/stable/) of `Agents.jl`.
- [sir-julia](https://github.com/epirecipes/sir-julia) : Various implementations of the classical SIR model in Julia.

## Notebook execution status

```{nb-exec-table}
```
