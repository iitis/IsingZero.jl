include("game.jl")
include("params.jl")

using AlphaZero: Scripts

experiment = Experiment("QuboZero", GameSpec(), params, Network, netparams, benchmark)

@show "train"
Scripts.train(experiment)
@show "explore"
Scripts.explore(experiment)


# TODO:
# 1. Make CUDA training work (scalar operation fails)
# 2. 