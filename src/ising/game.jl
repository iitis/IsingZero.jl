using AlphaZero
using LinearAlgebra
using Base: @kwdef
using DataStructures: CircularBuffer

const N = 10
const Q_upper_map = vec(triu(ones(Bool, (N, N))))

struct GameSpec <: GI.AbstractGameSpec
  tabu_buffer_size
  episode_length

  GameSpec() = begin

    # # solution obtained with bruteforce
    # energy_solution = -1172.0
    # solution = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1]

    tabu_buffer_size = 2
    episode_length = 7
    
    new(tabu_buffer_size, episode_length)
  end
end

energy(x, Q) = x' * Q * x

function random_x(n, Q, energy_solution)
  x = (abs.(rand(Int, n)) .% 2)
  while energy(x, Q) == energy_solution
    x = (abs.(rand(Int, n)) .% 2)
  end
  # TODO: if Q = 0, then we get stuck in an infinite loop :)
  x
end

linear_reward(E_initial, E_target, E_current) = ((E_initial - E_current) / (E_initial - E_target))
const AVAILABLE_ACTIONS = collect(1:N)

@kwdef mutable struct GameEnv <: GI.AbstractGameEnv
  x::Array{Float64,1}
  delta_E::Array{Float64,1}
  Q::Array{Float64, 2}
  energy_solution::Float64
  initial_energy::Float64 # Used to compute reward, if an improvement is found
  best_found_energy::Float64
  time::Int  # Count of the steps taken
  tabu_buffer::CircularBuffer{Int32}
end

GI.spec(::GameEnv) = GameSpec()

function GI.init(spec::GameSpec)
  Q, energy_solution = get_training_Q(Float64, N)
  x = random_x(N, Q, energy_solution)
  initial_energy = energy(x, Q) # changed only during reset / clone
  best_found_energy = copy(initial_energy)
  time = 0
  delta_E = compute_delta_E(Q, x)
  tabu_buffer = CircularBuffer{Int32}(spec.tabu_buffer_size)

  # TODO: add to input
  # 1. time since last improvement?
  # 3. ΔE_{t} - ΔE_{t-1}?: mówi o zmiennych, na które ma wpływ ostatnia zmiane
  ge = GameEnv(
    Q=Q,
    energy_solution=energy_solution,
    x=x,
    delta_E=delta_E,
    initial_energy=initial_energy,
    best_found_energy=best_found_energy,
    time=time,
    tabu_buffer=tabu_buffer,
  )
  return ge
end

function GI.set_state!(env::GameEnv, state)
  env.Q = deepcopy(state.Q)
  env.x = deepcopy(state.x)
  env.delta_E = deepcopy(state.delta_E)
  env.tabu_buffer = deepcopy(state.tabu_buffer)
  env.best_found_energy = deepcopy(state.best_found_energy)
  env.initial_energy = deepcopy(state.initial_energy)
end

GI.two_players(::GameSpec) = false


GI.actions(spec::GameSpec) = AVAILABLE_ACTIONS
function GI.clone(env::GameEnv)

  GameEnv(x=deepcopy(env.x), 
          Q=deepcopy(env.Q), 
          energy_solution=deepcopy(env.energy_solution), 
          delta_E=deepcopy(env.delta_E), 
          tabu_buffer=deepcopy(env.tabu_buffer),
          initial_energy=deepcopy(env.initial_energy),
          best_found_energy=deepcopy(env.best_found_energy),
          time=deepcopy(env.time))
end

history(::GameEnv) = nothing

#####
##### Defining game rules
#####

function GI.actions_mask(env::GameEnv)
  mask = BitVector([1 for _ in env.x])
  mask[env.tabu_buffer] .= 0
  return mask 
end

function GI.play!(env::GameEnv, a)
  env.tabu_buffer = deepcopy(env.tabu_buffer)
  push!(env.tabu_buffer, a)
  env.time += 1
  env.x = deepcopy(env.x)
  update_delta_E!(env.delta_E, env.Q, env.x, a)
  env.x[a] = 1 - env.x[a]
  new_energy = energy(env.x, env.Q)
  env.best_found_energy = min(env.best_found_energy, new_energy)
end

GI.current_state(env::GameEnv) = begin
  return (
    Q=env.Q,
    x=deepcopy(env.x),
    delta_E=deepcopy(env.delta_E),
    best_found_energy=deepcopy(env.best_found_energy),
    initial_energy=deepcopy(env.initial_energy),
    tabu_buffer=deepcopy(env.tabu_buffer)
  )
end

GI.white_playing(env::GameEnv) = true

#####
##### Reward shaping
#####

function GI.game_terminated(env::GameEnv)
  gspec = GI.spec(env)
  env.time == gspec.episode_length
end

function GI.white_reward(env::GameEnv)
  gspec = GI.spec(env)
  if env.time < gspec.episode_length
    return 0.0
  end

  if env.best_found_energy >= env.initial_energy
     return -1
  end
    
  r = linear_reward(env.initial_energy, env.energy_solution, env.best_found_energy)
  return r
end

#####
#  ML interface
#####


function GI.vectorize_state(::GameSpec, state)
  vector = []

  Q_elements = vec(state.Q)[Q_upper_map]

  found_improvement = state.best_found_energy < state.initial_energy
  buffer_clone = deepcopy(state.tabu_buffer)
  fill!(buffer_clone, -10) # -10 represents a value that was not filled in yet.
  sort!(buffer_clone) 
  state_contributors = [Q_elements, state.x,  found_improvement, buffer_clone] # state.delta_E,

  for item in state_contributors
    if item isa Number
      push!(vector, item)
    elseif item isa AbstractArray
      flattened = vec(item)
      append!(vector, flattened)
    else
      error("Unsupported type: $(typeof(item))")
    end
  end
  return Float32.(vector)
end


#####
#  User interface
#####

function GI.action_string(spec::GameSpec, a)
  return "Flipped idx $a"
end

function GI.parse_action(spec::GameSpec, s)
  return "$(parse(Int, s)) ??"
end

function GI.render(env::GameEnv)
  "RL.render"
  println("env.x = $(env.x); env.time = $(env.time); env.best_found_energy = $(env.best_found_energy); env.current = $(energy(env.x, Q)); env.tabu_buffer=$(env.tabu_buffer)")
end

function GI.read_state(spec::GameSpec)
  nothing
end

GI.heuristic_value(env::GameEnv) = 0.0
