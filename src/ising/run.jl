include("game.jl")
include("params.jl")

using AlphaZero: Scripts

experiment = Experiment("QuboZero", GameSpec(), params, Network, netparams, benchmark)

Scripts.train(experiment)
Scripts.train(experiment)
Scripts.train(experiment)
Scripts.train(experiment)
Scripts.train(experiment)
Scripts.explore(experiment)


# TODO:
# 1. Make CUDA training work (scalar operation fails)
# 2. 