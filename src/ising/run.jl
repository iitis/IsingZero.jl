using Dates
using UUIDs
ENV["GKSwstype"] = 100

include("dataset.jl")
include("gnngraph_operations.jl")
include("qubo_operations.jl")
include("game.jl")
include("params_game.jl")
include("params_gnn.jl")

using AlphaZero: Scripts



experiment_name = string(Dates.format(now(), "yyyymmddHHMMSS"), "-", uuid4())
name = "QuboZero:$(experiment_name)"
# name = "QuboZero:20231212115709-e7ab64df-a27d-48c5-81ac-1e0ea053dc7c"
experiment = Experiment(name, GameSpec(), params, GNN_Net, netparams, benchmark)

@show "test_game"
Scripts.test_game(experiment)
# @show "train"
Scripts.train(experiment)
# @show "explore"
# Scripts.play(experiment)


# 1. Jak nadpisać params.data w zapisie sesji, żeby zmienić parametry treningu?
# 2. 