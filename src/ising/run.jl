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


experiment_name = string(Dates.format(now(), "yyyy_mm_dd_HH:MM_SS__"), uuid4())
name = "QuboZero:$(experiment_name)"
# name = "QuboZero:20231212115709-e7ab64df-a27d-48c5-81ac-1e0ea053dc7c"
experiment = Experiment(name, GameSpec(), params, GNN_Net, netparams, benchmark)

# @show "test_game"
# Scripts.test_game(experiment)
# @show "train"
Scripts.train(experiment)
# @show "explore"
# Scripts.play(experiment) 


# 1. Jak nadpisać params.data w zapisie sesji, żeby zmienić parametry treningu?
# 2. 

# JEST:
# size(W) = (10, 64)
# size(X) = (10, 64)
# size(A) = (1, 64)
# size(P) = (1, 64)
# size(V) = (444, 64)

# Raczej powinno być:
# size(W) = (1, 64)
# size(X) = (444, 64)
# size(A) = (10, 64)
# size(P) = (10, 64)
# size(V) = (1, 64)