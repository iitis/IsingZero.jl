ENV["GKSwstype"] = 100

include("qubo_operations.jl")
include("game.jl")
include("params.jl")

using AlphaZero: Scripts

# Scripts.test_game("connect-four")

experiment = Experiment("QuboZero", GameSpec(), params, Network, netparams, benchmark)

# @show "test_game"
# Scripts.test_game(experiment)
@show "train"
Scripts.train(experiment)
@show "explore"
Scripts.explore(experiment)


