include("game.jl")
include("params.jl")

using AlphaZero: Scripts

experiment = Experiment(
  "AddToTen", GameSpec(), params, Network, netparams, benchmark)

Scripts.train(experiment)
Scripts.explore(experiment)
