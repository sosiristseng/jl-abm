#=====

# Schelling's segregation model

This example is taken from `Agents.jl` [tutorial](https://juliadynamics.github.io/Agents.jl/stable/examples/schelling/).

- Agents : They belong to one of two groups (0 or 1).
- Model : Each position of the grid can be occupied by at most one agent.
- For each step
  - If an agent has at least 3 neighbors belonging to the same group, then it is happy.
  - If an agent is unhappy, it keeps moving to new locations until it is happy.

To define an agent type, we should make a mutable struct derived from `AbstractAgent` with 2 mandatory fields:
- `id::Int` . The identifier number of the agent.
- `pos` . For agents on a 2D grid, the position field should be a tuple of 2 integers.

On top of that, we could define other properties for the agents.

```julia
mutable struct SchellingAgent <: AbstractAgent
    id::Int             # The identifier number of the agent
    pos::NTuple{2, Int} # The x, y location of the agent on a 2D grid
    mood::Bool          # whether the agent is happy in its position. (true = happy)
    group::Int          # The group of the agent, determines mood as it interacts with neighbors
end
```

=====#


#=====
## Setup

First, we create a 2D space with a Chebyshev metric. This leads to *8 neighboring positions* per position (except at the edges of the grid).

=====#

using Agents

## Creating a space
space = GridSpace((10, 10); periodic = false)

# We define the Agent type using the [`@agent`](https://juliadynamics.github.io/Agents.jl/stable/tutorial/#Agents.@agent) macro. Thus we don't have to setup the mandatory `id` and `pos` fields by ourselves. The relevant fileds are `mood` (whetehr the agent is happy or not) and `group` (which group the agent is on).

@agent SchellingAgent GridAgent{2} begin
    mood::Bool  ## True = happy
    group::Int  ## 0 or 1
end

# Parameter for the ABM
properties = Dict(:min_to_be_happy => 3)

# Define Schelling's model
schelling = ABM(SchellingAgent, space; properties)

# We setup the model using `initialize()` function to make the model easier to reproduce and change its parameter(s).

using Random ## for reproducibility in the RNG

function init_schelling(; numagents = 320, griddims = (20, 20), min_to_be_happy = 3, seed = 125)
    space = GridSpace(griddims, periodic = false)
    properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Random.MersenneTwister(seed)
    model = ABM(
        SchellingAgent, space;
        properties,
        rng,
        scheduler = Schedulers.randomly
    )

    ## Populate the model with agents, adding equal amount of the two types of agents at random positions in the model
    for n in 1:numagents
        agent = SchellingAgent(n, (1, 1), false, n < numagents / 2 ? 1 : 2)
        ## We don't need to set the starting position. Agents.jl chooses randomly for us.
        add_agent_single!(agent, model)
    end
    return model
end

# Finally, we define a step function `agent_step!()` to determine what happens to each agent. We use some built-in functions: `nearby_agents()` and `move_agent_single!()`.

function agent_step!(agent::SchellingAgent, model)
    minhappy = model.min_to_be_happy
    count_neighbors_same_group = 0
    ## For each neighbor, get group and compare to current agent's group
    ## and increment count_neighbors_same_group as appropriately.
    ## Here `nearby_agents` (with default arguments) will provide an iterator
    ## over the nearby agents one grid point away, which are at most 8.
    for neighbor in nearby_agents(agent, model)
        if agent.group == neighbor.group
            count_neighbors_same_group += 1
        end
    end
    if count_neighbors_same_group ≥ minhappy
        agent.mood = true ## The agent is happy
    else
        move_agent_single!(agent, model) ## Move the agent to a random position
    end
    return
end

# ## Stepping the model

model = init_schelling()

# The `step!()` function moves the model forward. The `run!()` function is similar to `step!()` but also collects data along the simulation.

# Move the model by one iteration
step!(model, agent_step!)

# Move the model by 3 iterations
step!(model, agent_step!, 3)

# ## Visualization

# The `abmplot()` function visulizes the simulation result. By default `Makie.jl` is used.

using InteractiveDynamics
using CairoMakie            ## Makie with the Cairo backend

# Some helper functions to identify agent groups.
groupcolor(a) = a.group == 1 ? :blue : :orange
groupmarker(a) = a.group == 1 ? :circle : :rect

# Plot the initial conditions of the model
model = init_schelling(griddims = (30, 30), numagents = 500)
figure, _ = abmplot(model; ac = groupcolor, am = groupmarker, as = 10)
figure

# Let's make an animation about the simulation.

model = init_schelling(griddims = (30, 30), numagents = 500)

abmvideo(
    "schelling.mp4", model, agent_step!;
    ac = groupcolor, am = groupmarker, as = 10,
    framerate = 4, frames = 20,
    title = "Schelling's segregation model"
)

# Jupyter notebooks only
using Base64

function display_mp4(filename)
    display("text/html", string("""<video autoplay controls><source src="data:video/x-m4v;base64,""",
        Base64.base64encode(open(read, filename)),"""" type="video/mp4"></video>"""))
end

display_mp4("schelling.mp4")

#====

## Data analysis

The `run!()` function runs simulation and collects data in the `DataFrame` format. The `adata` (aggregated data) keyword extacts information in the DataFrame.

====#

## aggregated data (adata). fields we want to extract
adata = [:pos, :mood, :group]

model = init_schelling()
data, _ = run!(model, agent_step!, 5; adata)
## print only the first 10 rows
data[1:10, :]

# aggregated data also accepts functions extracting information
x = (agent) -> agent.pos[1]
model = init_schelling()
adata = [x, :mood, :group]
data, _ = run!(model, agent_step!, 5; adata)
data[1:10, :]

#===
## Launching an interactive app

See [this section](https://juliadynamics.github.io/Agents.jl/stable/examples/schelling/#Launching-the-interactive-application-1) using `abm_data_exploration()` in the official tutorial.
===#

#===
## Saving the model states

- `AgentsIO.save_checkpoint()`
- `AgentsIO.load_checkpoint()`

https://juliadynamics.github.io/Agents.jl/stable/examples/schelling/#Saving/loading-the-model-state-1
===#
