using AlphaZero
using CommonRLInterface
using Crayons

const RL = CommonRLInterface

mutable struct World <: AbstractEnv
  state::Int # The sum of the previous steps
  time::Int  # Count of the steps taken
end

function World()
  World(0, 0)
end

RL.reset!(env::World) = begin 
  env.state = 0
  env.time = 0
end
RL.actions(env::World) = [0, 1, 2]
RL.observe(env::World) = env.state

RL.terminated(env::World) = env.time > 10

function RL.act!(env::World, a)
  env.state += a
  env.time += 1
  if env.state == 10
    return 5 / env.time
  elseif env.state > 10
    return -1
  end
  return 0
end

@provide RL.player(env::World) = 1 # A one player game
@provide RL.players(env::World) = [1]
@provide RL.observations(env::World) = (env.state, env.time)
@provide RL.clone(env::World) = World(env.state, env.state)
@provide RL.state(env::World) = env.state
@provide RL.setstate!(env::World, s) = (env.state = s)
@provide RL.valid_action_mask(env::World) = BitVector([1, 1, 1])

# Additional functions needed by AlphaZero.jl that are not present in 
# CommonRlInterface.jl. Here, we provide them by overriding some functions from
# GameInterface. An alternative would be to pass them as keyword arguments to
# CommonRLInterfaceWrapper.

function GI.render(env::World)
    println("env.state = $(env.state); env.time = $(env.time)")
end

function GI.vectorize_state(env::World, state)
  v = zeros(Float32, 2)
  v[1] = Float32(env.state)
  v[2] = Float32(env.time)
  return v
end

const action_names = ["Stay", "Add One", "Add Two"]

function GI.action_string(env::World, a)
  idx = findfirst(==(a), RL.actions(env))
  return isnothing(idx) ? "?" : action_names[idx]
end


function GI.parse_action(env::World, s)
  idx = findfirst(==(s), action_names)
  return isnothing(idx) ? nothing : RL.actions(env)[idx] # -1?
end

function GI.read_state(env::World)
    return (env.state, env.time)
end

GI.heuristic_value(::World) = 0.

GameSpec() = CommonRLInterfaceWrapper.Spec(World())