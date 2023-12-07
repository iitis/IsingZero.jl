using AlphaZero
using LinearAlgebra
using Base: @kwdef
using DataStructures: CircularBuffer

const N = 10
const Q = triu(reshape(Array(1:N^2), (10, 10))) .* repeat([1, -1], 5)
const Q_upper_map = vec(triu(ones(Bool, size(Q))))

struct GameSpec <: GI.AbstractGameSpec
  energy_solution
  solution
  tabu_buffer_size
  episode_length

  GameSpec() = begin
    # solution obtained with bruteforce
    energy_solution = -1172.0
    solution = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1]

    tabu_buffer_size = 4
    episode_length = 7
    
    new(energy_solution, solution, tabu_buffer_size, episode_length)
  end
end

energy(x, Q) = x' * Q * x
random_x(n) = (abs.(rand(Int, n)) .% 2)
reward(E_initial, E_target, E_current) = ((E_initial - E_current) / (E_initial - E_target))

@kwdef mutable struct GameEnv <: GI.AbstractGameEnv
  x::Array{Float64,1}
  delta_E::Array{Float64,1}
  solution::Array{Float64,1}
  energy_solution::Float64
  initial_energy::Float64 # Used to compute reward, if an improvement is found
  best_found_energy::Float64
  time::Int  # Count of the steps taken
  tabu_buffer::CircularBuffer{Int32}
  episode_length::Int
end

GI.spec(::GameEnv) = GameSpec()

function GI.init(spec::GameSpec)
  x = random_x(size(Q)[1])
  initial_energy = energy(x, Q) # changed only during reset / clone
  best_found_energy = copy(initial_energy)
  time = 0
  delta_E = compute_delta_E(Q, x)
  tabu_buffer = CircularBuffer{Int32}(spec.tabu_buffer_size)
  
  # TODO: add to input
  # 1. time since last improvement?
  # 3. ΔE_{t} - ΔE_{t-1}?: mówi o zmiennych, na które ma wpływ ostatnia zmiane
  ge = GameEnv(
    x=x,
    delta_E=delta_E,
    solution=spec.solution,
    energy_solution=spec.energy_solution,
    initial_energy=initial_energy,
    best_found_energy=best_found_energy,
    time=time,
    tabu_buffer=tabu_buffer,
    episode_length=spec.episode_length
  )
  return ge
end

function GI.set_state!(env::GameEnv, state)
  env.x = deepcopy(state.x)
  env.delta_E = deepcopy(state.delta_E)
  env.tabu_buffer = deepcopy(env.tabu_buffer)
  env.best_found_energy = deepcopy(state.best_found_energy)
  env.initial_energy = deepcopy(state.initial_energy)
end

GI.two_players(::GameSpec) = false

GI.actions(spec::GameSpec) = begin
  acts = [i for (i, _) in enumerate(spec.solution)]
  return acts
end
function GI.clone(env::GameEnv)

  GameEnv(x=deepcopy(env.x), delta_E=deepcopy(env.delta_E), tabu_buffer=deepcopy(env.tabu_buffer),
  solution=env.solution, energy_solution=env.energy_solution,
    initial_energy=deepcopy(env.initial_energy), best_found_energy=deepcopy(env.best_found_energy),
    time=deepcopy(env.time), episode_length=env.episode_length)
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
  "GI.play!"
  push!(env.tabu_buffer, a)
  env.time += 1
  env.x = deepcopy(env.x)
  update_delta_E!(env.delta_E, Q, env.x, a)
  env.x[a] = 1 - env.x[a]
  new_energy = energy(env.x, Q)
  env.best_found_energy = min(env.best_found_energy, new_energy)
end

GI.current_state(env::GameEnv) = begin
  return (
    Q=Q,
    x=deepcopy(env.x),
    delta_E=deepcopy(env.delta_E),
    best_found_energy=deepcopy(env.best_found_energy),
    initial_energy=deepcopy(env.initial_energy),
    # tabu_buffer=deepcopy(env.tabu_buffer)
  )
end

GI.white_playing(env::GameEnv) = true

#####
##### Reward shaping
#####

function GI.game_terminated(env::GameEnv)
  env.time == env.episode_length
end

function GI.white_reward(env::GameEnv)
  if env.time < env.episode_length
    return 0.0
  end

  # println("testing for improvement: $(env.best_found_energy) vs $(env.initial_energy)")
  if env.best_found_energy < env.initial_energy
    # println("found improvement: $(env.best_found_energy) < $(env.initial_energy)")
    r = reward(env.initial_energy, env.energy_solution, env.best_found_energy)
    if r > 1
      println("r > 1: env.best_found_energy=$(env.best_found_energy)")
    end
    return r
  else
    return -1
  end
end

#####
#  ML interface
#####


function GI.vectorize_state(::GameSpec, state)
  vector = []

  Q_elements = vec(state.Q)[Q_upper_map]

  found_improvement = state.best_found_energy < state.initial_energy
  # buffer_clone = deepcopy(state.tabu_buffer)
  # fill!(buffer_clone, -10) # -10 represents a value that was not filled in yet.
  # sort!(buffer_clone) 
  state_contributors = [Q_elements, state.x, state.delta_E, found_improvement] #  buffer_clone?

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
