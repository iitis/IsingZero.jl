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
  common = GNNChain(GCNConv(1 => hparams.hidden_dim1, relu),
    GCNConv(hparams.hidden_dim1 => hparams.hidden_dim2, relu),
    GCNConv(hparams.hidden_dim2 => hparams.hidden_dim2, relu),
    GlobalPool(mean)
  )
  vhead = Dense(hparams.hidden_dim2 => 1, tanh)
  # TODO: replace this layer with extra GCNConv and smart pooling to get rid of node_count
  phead = Dense(hparams.hidden_dim2 => gspec.problem_size)
  return GNN_Net(gspec, hparams, common, vhead, phead)
end

# function state_to_GNN_graph(state)
  
# end

function Network.forward(nn::GNN_Net, state)
  g = decode_gnngraph(state)

  c = nn.common(g)
  v = nn.vhead(c.gdata.u)
  p_linear = nn.phead(c.gdata.u)
  p = softmax(p_linear)
  # TODO do we need to reshape p, v? currently it is p_shape = (10, 1), v_shape = (1, 1). Seems fine.
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
