#===
# The spread of SARS-CoV-2 (Graph model)

[Source](https://juliadynamics.github.io/Agents.jl/stable/examples/sir/) from Agents.jl tutorial

Here we add one more category of individuals: those who are infected, but do not know it. Transmission rate for infected and diagnosed individuals is lower than infected and undetected.
===#

using Agents, Random, DataFrames, Graphs
using Distributions: Poisson, DiscreteNonParametric
using CairoMakie

# ## Define the Model

mutable struct PoorSoul <: AbstractAgent
    id::Int             ## Unique agent ID
    pos::Int            ## Which city
    days_infected::Int  ## number of days since is infected
    status::Symbol      ## S/I/R
end

#---

function make_SIRgraph(;
    Ns,                 ## Populations of the cities
    migration_rates,    ## Rate of people moving from one city to another
    β_und,              ## Transmission rate of infected but undetected
    β_det,              ## Transmission rate of infected and detected
    infection_period = 30,
    reinfection_probability = 0.05,
    detection_time = 14,
    death_rate = 0.02,
    Is = [zeros(Int, length(Ns) - 1)..., 1],  ## An array for initial number of infected but undetected people per city.
    seed = 2022,
)

    rng = MersenneTwister(seed)

    @assert length(Ns) == length(Is) == length(β_und) == length(β_det) == size(migration_rates, 1) "length of Ns, Is, and B, and number of rows/columns in migration_rates should be the same "
    @assert size(migration_rates, 1) == size(migration_rates, 2) "migration_rates rates should be a square matrix"

    ## Number of cities
    C = length(Ns)

    ## normalize migration_rates
    migration_rates_sum = sum(migration_rates, dims = 2)
    for c in 1:C
        migration_rates[c, :] ./= migration_rates_sum[c]
    end

    ## properties as a NamedTuple
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


    space = GraphSpace(complete_digraph(C))
    model = ABM(PoorSoul, space; properties, rng)

    ## Add initial susceptible individuals
    for city in 1:C, n in 1:Ns[city]
        ind = add_agent!(city, model, 0, :S)
    end
    ## add infected individuals
    for city in 1:C
        inds = ids_in_position(city, model)
        for n in 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I # Infected
            agent.days_infected = 1
        end
    end
    return model
end

#---

using LinearAlgebra: diagind

function make_SIRgraphParams(;
	C,
    max_travel_rate,
    infection_period = 30,
    reinfection_probability = 0.05,
    detection_time = 14,
    death_rate = 0.02,
    Is = [zeros(Int, C - 1)..., 1],
    seed = 2022,
)
	## For reproducibility
	Random.seed!(seed)

	## City population
    Ns = rand(50:5000, C)

	## Undetected transmission
    β_und = rand(0.3:0.02:0.6, C)

	## Detected transmission (set to 10% of undetected)
    β_det = β_und ./ 10

	## Migrate from city i to city j
	## People in small cities tend to migrate to bigger cities
	migration_rates = zeros(C, C)
    for c in 1:C, c2 in 1:C
        migration_rates[c, c2] = (Ns[c] + Ns[c2]) / Ns[c]
    end

	## Normalize migration rates
	maxM = maximum(migration_rates)
    migration_rates = (migration_rates .* max_travel_rate) ./ maxM

	## Migrate to self = 1
    migration_rates[diagind(migration_rates)] .= 1.0

	return (; Ns,
        β_und,
        β_det,
        migration_rates,
        infection_period,
        reinfection_probability,
        detection_time,
        death_rate,
        Is)
end

#--

SIRgraphparams = make_SIRgraphParams(C = 8, max_travel_rate = 0.01)

#--

# Stepping function in the SIR Agent-based model

function migrate!(agent::PoorSoul, model)
    pid = agent.pos
    d = DiscreteNonParametric(1:(model.C), model.migration_rates[pid, :])
    m = rand(model.rng, d)
    if m ≠ pid
        move_agent!(agent, m, model)
    end
end

function transmit!(agent::PoorSoul, model)
    agent.status == :S && return
    rate = if agent.days_infected < model.detection_time
        model.β_und[agent.pos]
    else
        model.β_det[agent.pos]
    end

    d = Poisson(rate)
    n = rand(model.rng, d)
    n == 0 && return

    for contactID in ids_in_position(agent, model)
        contact = model[contactID]
        if contact.status == :S ||
           (contact.status == :R && rand(model.rng) ≤ model.reinfection_probability)
            contact.status = :I
            n -= 1
            n == 0 && return
        end
    end
end

# Count infected days of the agent
update!(agent::PoorSoul, model) = agent.status == :I && (agent.days_infected += 1)

function recover_or_die!(agent::PoorSoul, model)
    if agent.days_infected ≥ model.infection_period
        if rand(model.rng) ≤ model.death_rate
            kill_agent!(agent, model)
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end

function agent_step!(agent::PoorSoul, model)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end

#---

model = make_SIRgraph(; SIRgraphparams...)

#===
## Animation

At the moment [abmplot](https://juliadynamics.github.io/Agents.jl/stable/agents_visualizations/#InteractiveDynamics.abmplot) does not plot `GraphSpace`s, but we can still utilize the [ABMObservable](https://juliadynamics.github.io/Agents.jl/stable/agents_visualizations/#InteractiveDynamics.ABMObservable). We do not need to collect data here, only the current status of the model will be used in visualization.
===#

using InteractiveDynamics
using CairoMakie

## Observable: The quantity that updates dynamically and interactively
abmobs = ABMObservable(model; agent_step!)

infected_fraction(m, x) = count(m[id].status == :I for id in x) / length(x)
infected_fractions(m) = [infected_fraction(m, ids_in_position(p, m)) for p in positions(m)]

# Connect (lift) model obervable to fracs, color, and the title.
fracs = lift(infected_fractions, abmobs.model)
color = lift(fs -> [cgrad(:inferno)[f] for f in fs], fracs)
title = lift(
    (s, m) -> "step = $(s), infected = $(round(Int, 100infected_fraction(m, allids(m))))%",
    abmobs.s, abmobs.model
)

fig = Figure(resolution = (600, 400))
ax = Axis(fig[1, 1]; title, xlabel = "City", ylabel = "Population")
barplot!(ax, model.Ns; strokecolor = :black, strokewidth = 1, color)
fig

record(fig, "covid_evolution.mp4"; framerate = 5) do io
    for j in 1:30
        recordframe!(io)
        Agents.step!(abmobs, 1)
    end
    recordframe!(io)
end

using Base64

function display_mp4(filename)
    display("text/html", string("""<video autoplay controls><source src="data:video/x-m4v;base64,""",
        Base64.base64encode(open(read, filename)),"""" type="video/mp4"></video>"""))
end

display_mp4("covid_evolution.mp4")

## Data Collection

# Helper functions
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x)

#---
model = make_SIRgraph(; SIRgraphparams...)

to_collect = [(:status, f) for f in (infected, recovered, length)]
data, _ = run!(model, agent_step!, 100; adata = to_collect)
data[1:10, :]

N = sum(model.Ns) # Total initial population
x = data.step
fig = Figure(resolution = (600, 400))
ax = fig[1, 1] = Axis(fig, xlabel = "steps", ylabel = "log10(count)")
li = lines!(ax, x, log10.(data[:, aggname(:status, infected)]), color = :blue)
lr = lines!(ax, x, log10.(data[:, aggname(:status, recovered)]), color = :red)
dead = log10.(N .- data[:, aggname(:status, length)])
ld = lines!(ax, x, dead, color = :green)
fig[1, 2] = Legend(fig, [li, lr, ld], ["infected", "recovered", "dead"])
fig

# The exponential growth is clearly visible since the logarithm of the number of infected increases linearly, until everyone is infected.
