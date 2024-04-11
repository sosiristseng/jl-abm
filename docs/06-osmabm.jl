#===
# Open street map example

The zombie outbreak model showcases an ABM running on a map, using `OpenStreetMapSpace`.
===#

using Agents
using Random
using CairoMakie
using OSMMakie
CairoMakie.activate!(px_per_unit = 1.0)

# The helper function `display_mp4()` displays mp4 files in Jupyter notebooks
using Base64
function display_mp4(filename)
    display("text/html", string("""<video autoplay controls><source src="data:video/x-m4v;base64,""",
        Base64.base64encode(open(read, filename)),"""" type="video/mp4"></video>"""))
end

# Agents
@agent struct Zombie(OSMAgent)
    infected::Bool
    speed::Float64
end

# Model factory
function initialise_zombies(; seed = 1234)
    map_path = OSM.test_map()
    properties = Dict(:dt => 1 / 60)

    model = StandardABM(
        Zombie,
        OpenStreetMapSpace(map_path);
        agent_step! = zombie_step!,
        properties = properties,
        rng = Random.MersenneTwister(seed)
    )

    for id in 1:100
        start = random_position(model) ## At an intersection
        speed = rand(abmrng(model)) * 5.0 + 2.0 ## Random speed from 2-7kmph
        human = add_agent!(start, Zombie, model, false, speed)
        OSM.plan_random_route!(human, model; limit = 50) ## try 50 times to find a random route
    end

    ## We'll add patient zero at a specific (longitude, latitude)
    start = OSM.nearest_road((9.9351811, 51.5328328), model)
    finish = OSM.nearest_node((9.945125635913511, 51.530876112711745), model)

    speed = rand(abmrng(model)) * 5.0 + 2.0 ## Random speed from 2-7kmph
    zombie = add_agent!(start, model, true, speed)
    plan_route!(zombie, finish, model)
    return model
end

# Stepping function
function zombie_step!(agent, model)
    ## Each agent will progress along their route
    ## Keep track of distance left to move this step, in case the agent reaches its
    ## destination early
    distance_left = move_along_route!(agent, model, agent.speed * model.dt)

    if is_stationary(agent, model) && rand(abmrng(model)) < 0.1
        ## When stationary, give the agent a 10% chance of going somewhere else
        OSM.plan_random_route!(agent, model; limit = 50)
        ## Start on new route, moving the remaining distance
        move_along_route!(agent, model, distance_left)
    end

    ## Agents will be infected if they get too close (within 10m) to a zombie.
    if agent.infected
        map(i -> model[i].infected = true, nearby_ids(agent, model, 0.01))
    end
    return
end

# ## Animation
zombie_color(agent) = agent.infected ? :green : :black
zombie_size(agent) = agent.infected ? 10 : 8
zombies = initialise_zombies()

abmvideo("outbreak.mp4", zombies;
    title = "Zombie outbreak", framerate = 15, frames = 200,
    agent_color = zombie_color, agent_size = zombie_size
)

display_mp4("outbreak.mp4")
