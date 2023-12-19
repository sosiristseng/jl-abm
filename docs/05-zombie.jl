# # Zombie Outbreak: An OpenStreetMap Example
# Source: https://juliadynamics.github.io/Agents.jl/stable/examples/zombies/

using Agents
using Random

#===
Define zombies/humans to be Openstreetmap (OSM) agents.

The following code, which uses `@agent` macro, is equivalent to

```julia
mutable struct Zombie <: AbstractAgent
    id::Int
    pos::Tuple{Int, Int, Float64}
    infected::Bool
    speed::Float64
end
```
===#
@agent Zombie OSMAgent begin
    infected::Bool
    speed::Float64
end

#---
function init_zombie(; seed = 1234)
    map_path = OSM.test_map()
    properties = Dict(:dt => 1 / 60)
    model = ABM(
        Zombie,
        OpenStreetMapSpace(map_path);
        properties = properties,
        rng = Random.MersenneTwister(seed)
    )

    for id in 1:100
        start = random_position(model) ## At an intersection
        speed = rand(model.rng) * 5.0 + 2.0 ## Random speed from 2-7kmph
        human = Zombie(id, start, false, speed)
        add_agent_pos!(human, model)
        OSM.plan_random_route!(human, model; limit = 50) ## try 50 times to find a random route
    end

    ## We'll add patient zero at a specific (longitude, latitude)
    start = OSM.nearest_road((9.9351811, 51.5328328), model)
    finish = OSM.nearest_node((9.945125635913511, 51.530876112711745), model)

    speed = rand(model.rng) * 5.0 + 2.0 ## Random speed from 2-7 kmph
    zombie = add_agent!(start, model, true, speed)
    plan_route!(zombie, finish, model)
    ## This function call creates & adds an agent, see `add_agent!`
    return model
end

#---
function zombie_step!(agent::Zombie, model)
    ## Each agent will progress along their route
    ## Keep track of distance left to move this step, in case the agent reaches its
    ## destination early
    distance_left = move_along_route!(agent, model, agent.speed * model.dt)

    if is_stationary(agent, model) && rand(model.rng) < 0.1
        ## When stationary, give the agent a 10% chance of going somewhere else
        OSM.plan_random_route!(agent, model; limit = 50)
        ## Start on new route, moving the remaining distance
        move_along_route!(agent, model, distance_left)
    end

    if agent.infected
        ## Agents will be infected if they get too close (within 10m) to a zombie.
        map(i -> model[i].infected = true, nearby_ids(agent, model, 0.01))
    end
    return
end

# ## Visulization

using CairoMakie
using OSMMakie

zombie_color(agent::Zombie) = agent.infected ? :green : :black
zombie_size(agent::Zombie) = agent.infected ? 10 : 8

model = init_zombie()

Agents.abmvideo(
    "outbreak.mp4",
    model, zombie_step!;
    title = "Zombie outbreak",
    framerate = 15, frames = 200,
    as=zombie_size, ac=zombie_color
)

using Base64
function display_mp4(filename)
    display("text/html", string("""<video autoplay controls><source src="data:video/x-m4v;base64,""",
        Base64.base64encode(open(read, filename)),"""" type="video/mp4"></video>"""))
end

display_mp4("outbreak.mp4")
