include("game.jl")
include("params.jl")

using AlphaZero: Scripts

experiment = Experiment("QuboZero", GameSpec(), params, Network, netparams, benchmark)

Scripts.train(experiment)
Scripts.explore(experiment)
