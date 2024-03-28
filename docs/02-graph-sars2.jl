#===
# The spread of SARS-CoV-2 in a Graph model

From Agents.jl tutorial : https://juliadynamics.github.io/Agents.jl/stable/examples/sir/

Here we add one more category of individuals: those who are infected, but do not know it. Transmission rate for infected and diagnosed individuals is lower than infected and undetected.

## Define the Model
===#

using Agents, Random
using Agents.DataFrames, Agents.Graphs
using StatsBase: sample, Weights
using CairoMakie

# Agent
@agent struct PoorSoul(GraphAgent)
    days_infected::Int  ## number of days since is infected
    status::Symbol      ## 1: S, 2: I, 3:R
end

# Build the ABM
function model_initiation(;
    Ns,                 ## Populations of the cities
    migration_rates,    ## Rate of people moving from one city to another
    β_und,              ## Transmission rate of infected but undetected
    β_det,              ## Transmission rate of infected and detected
    infection_period = 30,
    reinfection_probability = 0.05,
    detection_time = 14,
    death_rate = 0.02,
    Is = [zeros(Int, length(Ns) - 1)..., 1], ## An array for initial number of infected but undetected people per city.
    seed = 2024,
)

    rng = Xoshiro(seed)

    @assert length(Ns) ==
    length(Is) ==
    length(β_und) ==
    length(β_det) ==
    size(migration_rates, 1) "length of Ns, Is, and B, and number of rows/columns in migration_rates should be the same "
    @assert size(migration_rates, 1) == size(migration_rates, 2) "migration_rates rates should be a square matrix"

    ## Number of cities
    C = length(Ns)

    # normalize migration rates
    migration_rates_sum = sum(migration_rates, dims = 2)
    for c in 1:C
        migration_rates[c, :] ./= migration_rates_sum[c]
    end

    ## properties parameter set

    properties = @dict(
        Ns,
        Is,
        β_und,
        β_det,
        β_det,
        migration_rates,
        infection_period,
        infection_period,
        reinfection_probability,
        detection_time,
        C,
        death_rate
    )

    space = GraphSpace(complete_graph(C))
    model = StandardABM(PoorSoul, space; agent_step!, properties, rng)

    # Add initial individuals
    for city in 1:C, n in 1:Ns[city]
        ind = add_agent!(city, model, 0, :S) # Susceptible
    end
    # add infected individuals
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
