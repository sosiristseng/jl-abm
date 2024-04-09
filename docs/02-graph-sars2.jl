#===
# The spread of SARS-CoV-2 (Graph model)

[Source](https://juliadynamics.github.io/Agents.jl/stable/examples/sir/) from Agents.jl examples

Here we add one more category of individuals: those who are infected, but do not know it. Transmission rate for infected and diagnosed individuals is lower than infected and undetected.
===#

using Agents, Random
using Agents.DataFrames, Agents.Graphs
using StatsBase: sample, Weights
using CairoMakie
using LinearAlgebra: diagind

# ## Define the Model
@agent struct PoorSoul(GraphAgent)
    days_infected::Int  ## number of days since is infected
    status::Symbol  ## S/I/R
end

# Model factory function
function make_model(;
    Ns,
    migration_rates,
    β_und,
    β_det,
    infection_period = 30,
    reinfection_probability = 0.05,
    detection_time = 14,
    death_rate = 0.02,
    Is = [zeros(Int, length(Ns) - 1)..., 1],
    seed = 2024,
)

    rng = Xoshiro(seed)
    @assert length(Ns) ==
    length(Is) ==
    length(β_und) ==
    length(β_det) ==
    size(migration_rates, 1) "length of Ns, Is, and B, and number of rows/columns in migration_rates should be the same "
    @assert size(migration_rates, 1) == size(migration_rates, 2) "migration_rates rates should be a square matrix"

    C = length(Ns) ## Number of cities

    ## normalize migration_rates
    migration_rates_sum = sum(migration_rates, dims = 2)
    for c in 1:C
        migration_rates[c, :] ./= migration_rates_sum[c]
    end

    properties = (;
        Ns,
        Is,
        β_und,
        β_det,
        migration_rates,
        infection_period,
        reinfection_probability,
        detection_time,
        C,
        death_rate
    )

    space = GraphSpace(complete_graph(C))
    model = StandardABM(PoorSoul, space; agent_step!, properties, rng)

    ## Add initial individuals
    for city in 1:C, n in 1:Ns[city]
        ind = add_agent!(city, model, 0, :S) ## Susceptible
    end

    ## add infected individuals
    for city in 1:C
        inds = ids_in_position(city, model)
        for n in 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I ## Set infected individual
            agent.days_infected = 1
        end
    end
    return model
end

# Parameter factory function
function create_params(;
    C,
    max_travel_rate,
    infection_period = 30,
    reinfection_probability = 0.05,
    detection_time = 14,
    death_rate = 0.02,
    Is = [zeros(Int, C - 1)..., 1],
    seed = 2024,
)
    rng = Xoshiro(seed)
    Ns = rand(rng, 50:5000, C) ## City population
    β_und = rand(rng, 0.3:0.02:0.6, C) ## Undetected transmission
    β_det = β_und ./ 10 ## Detected transmission (set to 10% of undetected)

    ## Migrate from city i to city j
	## People in small cities tend to migrate to bigger cities
    migration_rates = zeros(C, C)
    for c in 1:C
        for c2 in 1:C
            migration_rates[c, c2] = (Ns[c] + Ns[c2]) / Ns[c]
        end
    end

    ## Normalize migration rates
    maxM = maximum(migration_rates)
    migration_rates = (migration_rates .* max_travel_rate) ./ maxM
    ## Migrate to self = 1
    migration_rates[diagind(migration_rates)] .= 1.0

    params = (;
        Ns,
        β_und,
        β_det,
        migration_rates,
        infection_period,
        reinfection_probability,
        detection_time,
        death_rate,
        Is
    )

    return params
end

# ## Stepping functions
function agent_step!(agent, model)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
    return nothing
end

function migrate!(agent, model)
    pid = agent.pos
    ## StatsBase.sample
    m = sample(abmrng(model), 1:(model.C), Weights(model.migration_rates[pid, :]))
    if m ≠ pid
        move_agent!(agent, m, model)
    end
    return nothing
end

function transmit!(agent, model)
    agent.status == :S && return
    rate = if agent.days_infected < model.detection_time
        model.β_und[agent.pos]
    else
        model.β_det[agent.pos]
    end

    rng = abmrng(model)
    n = rate * abs(randn(rng))
    n <= 0 && return

    for contactID in ids_in_position(agent, model)
        contact = model[contactID]
        if contact.status == :S ||
           (contact.status == :R && rand(rng) ≤ model.reinfection_probability)
            contact.status = :I
            n -= 1
            n <= 0 && return
        end
    end
    return nothing
end

update!(agent, model) = agent.status == :I && (agent.days_infected += 1)

function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period
        if rand(abmrng(model)) ≤ model.death_rate
            remove_agent!(agent, model)
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
    return nothing
end

# Initialize the model
params = create_params(C = 10, max_travel_rate = 0.01)
model = make_model(; params...)

# ## Animation
# Observable: The quantity that updates dynamically and interactively Makie plots.

abmobs = ABMObservable(model)

infected_fraction(m, x) = count(m[id].status == :I for id in x) / length(x)
infected_fractions(m) = [infected_fraction(m, ids_in_position(p, m)) for p in positions(m)]

# Connect (lift) observables to model stats
fracs = lift(infected_fractions, abmobs.model)
color = lift(fs -> [cgrad(:inferno)[f] for f in fs], fracs)
title = lift(
    (m) -> "step = $(abmtime(m)), infected = $(round(Int, 100infected_fraction(m, allids(m))))%",
    abmobs.model
)

# Figure
fig = Figure(size = (600, 400))
ax = Axis(fig[1, 1]; title, xlabel = "City", ylabel = "Population")
barplot!(ax, model.Ns; strokecolor = :black, strokewidth = 1, color)
fig

# Animation
record(fig, "covid_evolution.mp4"; framerate = 5) do io
    for j in 1:30
        recordframe!(io)
        Agents.step!(abmobs, 1)
    end
    recordframe!(io)
end

# Display video files in Jupyter Notebooks.
using Base64

function display_mp4(filename)
    display("text/html", string("""<video autoplay controls><source src="data:video/x-m4v;base64,""",
        Base64.base64encode(open(read, filename)),"""" type="video/mp4"></video>"""))
end

display_mp4("covid_evolution.mp4")

# ## Data Collection
# Define helper functions
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x)

# Run the model and collect the data
model = make_model(; params...)
to_collect = [(:status, f) for f in (infected, recovered, length)]
data, _ = run!(model, 100; adata = to_collect)
data[1:10, :]

# ## Visualize
N = sum(model.Ns) ## Total initial population
fig = Figure(size = (600, 400))
ax = fig[1, 1] = Axis(fig, xlabel = "steps", ylabel = "log10(count)")
li = lines!(ax, data.time, log10.(data[:, dataname((:status, infected))]), color = :blue)
lr = lines!(ax, data.time, log10.(data[:, dataname((:status, recovered))]), color = :red)
dead = log10.(N .- data[:, dataname((:status, length))])
ld = lines!(ax, data.time, dead, color = :green)
Legend(fig[1, 2], [li, lr, ld], ["infected", "recovered", "dead"])
fig
