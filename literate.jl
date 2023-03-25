using Distributed

# For all processes
@everywhere begin
    import Pkg
    Pkg.activate(@__DIR__)

    using Literate
    config = Dict("mdstrings" => true)
end

folder = joinpath(@__DIR__, "docs")

nbs = (
    "01-schelling.jl"
    "02-graph-sars2.jl",
    "03-flocking.jl",
    "04-social-distancing.jl",
    "05-zombie.jl",
)

ts = pmap(nbs; on_error=identity) do nb
    @elapsed Literate.notebook(joinpath(folder, nb), folder; config)
end

for (nb, t) in zip(nbs, ts)
    println(nb, " elapsed/error: ", t)
end
