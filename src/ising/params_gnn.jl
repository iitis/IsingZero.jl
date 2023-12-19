# Network = NetLib.SimpleNet

# netparams = NetLib.SimpleNetHP(
#   width=512,
#   depth_common=5,
#   use_batch_norm=true,
#   depth_phead=2,
#   depth_vhead=2
#   )


@kwdef struct GNN_HP
    hidden_feature_dim::Int = 8

end


"""
    GNN_HP

Hyperparameters for the GNN architecture.

| Parameter                     | Description                                  |
|:------------------------------|:---------------------------------------------|
| `hidden_feature_dim :: Int`   | TODO                                         |
"""

"""
    SimpleNet <: TwoHeadNetwork

A simple two-headed architecture with only dense layers.
"""


mutable struct GNN_Net <: TwoHeadNetwork
  gspec
  hyper
  common
  vhead
  phead
end

function GNN_Net(gspec::AbstractGameSpec, hyper::GNN_HP)
  bnmom = hyper.batch_norm_momentum
  function make_dense(indim, outdim)
    if hyper.use_batch_norm
      Chain(
        Dense(indim, outdim),
        BatchNorm(outdim, relu, momentum=bnmom))
    else
      Dense(indim, outdim, relu)
    end
  end
  indim = prod(GI.state_dim(gspec))
  outdim = GI.num_actions(gspec)
  hsize = hyper.width
  hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth]
  common = Chain(
    flatten,
    make_dense(indim, hsize),
    hlayers(hyper.depth_common)...)
  vhead = Chain(
    hlayers(hyper.depth_vhead)...,
    Dense(hsize, 1, tanh))
  phead = Chain(
    hlayers(hyper.depth_phead)...,
    Dense(hsize, outdim),
    softmax)
  SimpleNet(gspec, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{SimpleNet}) = SimpleNetHP

function Base.copy(nn::SimpleNet)
  return SimpleNet(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end