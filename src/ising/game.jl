using AlphaZero
using CommonRLInterface
using Crayons
using LinearAlgebra

const RL = CommonRLInterface

const EPISODE_LENGTH = 10

energy(x, Q) = x' * Q * x
random_x(n) = Array{Real}(abs.(rand(Int, n)) .% 2)

mutable struct World <: AbstractEnv
  Q::Array{Real,2} # The sum of the previous steps
  x::Array{Real,1}

  action_history::Vector{Int}
  energy_history::Vector{Real}

  time::Int  # Count of the steps taken
end

# TODO: introduce additional structures to organize the World

function World()
  Q = triu(reshape(Array(1:25), (5, 5))) .* [1.0, -1.0, 1.0, -1.0, 1.0] # Just a temporary, random QUBO
  x = random_x(size(Q)[1]) # state
  e = energy(x, Q)
  time = 0
  return World(Q, x, [], [e], time)
end

RL.reset!(env::World) = begin
  env.x = random_x(size(env.Q)[1])
  env.action_history = []
  env.energy_history = [energy(env.x, env.Q)]
  env.time = 0
end

RL.actions(env::World) = collect(1:length(env.x))
RL.observe(env::World) = (env.Q, env.x, env.time)

RL.terminated(env::World) = env.time > EPISODE_LENGTH

function RL.act!(env::World, a)
  env.x[a] = 1 - env.x[a]
  push!(env.action_history, a)
  push!(env.energy_history, energy(env.x, env.Q))

  env.time += 1
  if env.time == EPISODE_LENGTH
    if any(env.energy_history .> env.energy_history[begin])
      return 1
    else
      return -1
    end
  end
  return 0
end

@provide RL.player(env::World) = 1 # A one player game
@provide RL.players(env::World) = [1]
@provide RL.observations(env::World) = (env.x, env.time)
@provide RL.clone(env::World) = World(env.Q, env.x, env.action_history, env.energy_history, env.time)
@provide RL.state(env::World) = (env.x, env.Q, env.energy_history[end])
@provide RL.valid_action_mask(env::World) = begin
  # TODO: This is where we limit flipping of the same index
  return BitVector([1 for _ in env.x])
end
@provide RL.setstate!(env::World, x) = begin
  env.x = x
  env.energy_history(energy(env.x, env.Q))
end

# Additional functions needed by AlphaZero.jl that are not present in 
# CommonRlInterface.jl. Here, we provide them by overriding some functions from
# GameInterface. An alternative would be to pass them as keyword arguments to
# CommonRLInterfaceWrapper.

function GI.render(env::World)
  println("env.x = $(env.x); env.time = $(env.time); env.energy_history = $(env.energy_history)")
end

function GI.vectorize_state(env::World, state)
  vector = []
  for item in state
      if item isa Number
          push!(vector, item)
      elseif item isa AbstractArray
          flattened = vec(item)
          append!(vector, flattened)
      else
          error("Unsupported type: $(typeof(item))")
      end
  end
  return vector
end

function GI.action_string(env::World, a)
  return "Flipped $a"
end

function GI.parse_action(env::World, s)
  idx = findfirst(==(s), action_names)
  return isnothing(idx) ? nothing : RL.actions(env)[idx]
end

function GI.read_state(env::World)
  return (env.state, env.time)
end

GI.heuristic_value(::World) = 0.0

GameSpec() = CommonRLInterfaceWrapper.Spec(World())