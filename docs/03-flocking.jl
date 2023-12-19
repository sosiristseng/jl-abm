#===
# Flocking of birds

The flock model illustrates how flocking behavior can emerge when each bird follows three simple rules:

- maintain a minimum distance from other birds to avoid collision
- fly towards the average position of neighbors
- fly in the average direction of neighbors

## Model setup
===#

using Agents
using Random
using LinearAlgebra

@agent Bird ContinuousAgent{2} begin
    speed::Float64
    cohere_factor::Float64
    separation::Float64
    separate_factor::Float64
    match_factor::Float64
    visual_distance::Float64
end

#---
function init_flocking(;
    n_birds = 100,
    speed = 1.0,
    cohere_factor = 0.25,
    separation = 4.0,
    separate_factor = 0.25,
    match_factor = 0.01,
    visual_distance = 5.0,
    extent = (100, 100),
    spacing = visual_distance / 1.5,
)
    space2d = ContinuousSpace(extent; spacing)
    model = ABM(Bird, space2d, scheduler = Schedulers.Randomly())
    for _ in 1:n_birds
        vel = Tuple(rand(model.rng, 2) * 2 .- 1)
        add_agent!(
            model,
            vel,
            speed,
            cohere_factor,
            separation,
            separate_factor,
            match_factor,
            visual_distance,
        )
    end
    return model
end

#---
function agent_step!(bird::Bird, model)
    ## Obtain the ids of neighbors within the bird's visual distance
    neighbor_ids = nearby_ids(bird, model, bird.visual_distance)
    N = 0
    match = separate = cohere = (0.0, 0.0)
    ## Calculate behaviour properties based on neighbors
    for id in neighbor_ids
        N += 1
        neighbor = model[id].pos
        heading = neighbor .- bird.pos

        ## `cohere` computes the average position of neighboring birds
        cohere = cohere .+ heading
        if edistance(bird.pos, neighbor, model) < bird.separation
            ## `separate` repels the bird away from neighboring birds
            separate = separate .- heading
        end
        ## `match` computes the average trajectory of neighboring birds
        match = match .+ model[id].vel
    end
    N = max(N, 1)
    ## Normalise results based on model input and neighbor count
    cohere = cohere ./ N .* bird.cohere_factor
    separate = separate ./ N .* bird.separate_factor
    match = match ./ N .* bird.match_factor
    ## Compute velocity based on rules defined above
    bird.vel = (bird.vel .+ cohere .+ separate .+ match) ./ 2
    bird.vel = bird.vel ./ norm(bird.vel)
    ## Move bird according to new velocity and speed
    move_agent!(bird, model, bird.speed)
end

# ## Visualization
using CairoMakie

const bird_polygon = Makie.Polygon(Point2f[(-0.5, -0.5), (1, 0), (-0.5, 0.5)])
function bird_marker(b::Bird)
    φ = atan(b.vel[2], b.vel[1])
    rotate_polygon(bird_polygon, φ)
end

model = init_flocking()
figure, = Agents.abmplot(model; am = bird_marker)
figure

# ## Animation
model = init_flocking()

Agents.abmvideo(
    "flocking.mp4", model, agent_step!;
    am = bird_marker,
    framerate = 20,
    frames = 200,
    title = "Flocking"
)

using Base64

function display_mp4(filename)
    display("text/html", string("""<video autoplay controls><source src="data:video/x-m4v;base64,""",
        Base64.base64encode(open(read, filename)),"""" type="video/mp4"></video>"""))
end

display_mp4("flocking.mp4")
