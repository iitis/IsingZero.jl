using Base: load_InteractiveUtils
using GraphNeuralNetworks
using AlphaZero: NetLib
using Statistics: mean


mutable struct GNN_Net <: NetLib.TwoHeadNetwork
  gspec
  hyper
  common
  vhead
  phead
end

@kwdef struct GNN_HP
  input_node_feature_count::Int = 4
  edge_feature_dim::Int = 1
  hidden_dim1::Int = 320
  hidden_dim2::Int = 10
end

# hidden_dim1::Int = 8
# hidden_dim2::Int = 10
# state = GNNGraph(320, 1760) with x: 1×320 data
# DimensionMismatch: A has dimensions (8,10) but B has dimensions (1,320)

# hidden_dim1::Int = 15
# hidden_dim2::Int = 10

function GNN_Net(gspec::AbstractGameSpec, hparams::GNN_HP)
  common = GNNChain(
    GATConv((hparams.input_node_feature_count, hparams.edge_feature_dim) => hparams.hidden_dim1, relu, add_self_loops=false),
    GATConv((hparams.hidden_dim1, hparams.edge_feature_dim) => hparams.hidden_dim2, relu, add_self_loops=false),
    GATConv((hparams.hidden_dim2, hparams.edge_feature_dim) => hparams.hidden_dim2, relu, add_self_loops=false),
    GlobalPool(mean)
  )
  vhead = Dense(hparams.hidden_dim2 => 1, tanh)
  # TODO: replace this layer with extra GCNConv and smart pooling to get rid of node_count
  phead = Dense(hparams.hidden_dim2 => gspec.problem_size)
  return GNN_Net(gspec, hparams, common, vhead, phead)
end


function convert_input(nn::GNN_Net, input::Union{Tuple, NamedTuple})
    @show input
    return input
end

function Network.forward(nn::GNN_Net, state)
  # @show size(state)
  batch_size = size(state, 2)
  graphs = [decode_gnngraph(state[:, i]) for i in 1:batch_size]
  loader = Flux.DataLoader(graphs, shuffle=false, collate=true, batchsize=batch_size)
  pred_batch = first(loader) 
  c = nn.common(pred_batch)
  v = nn.vhead(c.gdata.u)
  p_linear = nn.phead(c.gdata.u)
  p = softmax(p_linear)
  return (p, v)
end


Network.HyperParams(::Type{GNN_Net}) = GNN_HP

function Base.copy(nn::GNN_Net)
  return GNN_Net(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end


netparams = GNN_HP()

Network.on_gpu(nn::GNN_Net) = false
