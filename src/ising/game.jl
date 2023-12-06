using AlphaZero
using LinearAlgebra
using Base: @kwdef

#####
##### Game Specifications
#####

const EPISODE_LENGTH = 5

struct GameSpec <: GI.AbstractGameSpec
  Q::Matrix{Float64}
  energy_solution::Float64
  solution::Vector{Float64}

  GameSpec() = begin
    N = 10
    Q = Float64.(triu(reshape(Array(1:N^2), (N, N))) .* repeat([1, -1], Int(N // 2))) # Just a temporary, random QUBO

    # obrained with bruteforce
    energy_solution = -1172.0
    solution = Float64[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    new(Q, energy_solution, solution)
  end

end

GI.two_players(::GameSpec) = false
GI.actions(spec::GameSpec) = [i for (i, _) in enumerate(spec.solution)]


#####
##### ML interface
#####

function GI.vectorize_state(::GameSpec, state)
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
  return Float32.(vector)
end

#####
##### Game Environment
#####

energy(x, Q) = x' * Q * x
random_x(n) = rand([0., 1.], n)
reward(E_initial, E_target, E_current) = ((E_initial - E_current) / (E_initial - E_target))

@kwdef mutable struct GameEnv{T} <: GI.AbstractGameEnv
  Q::Matrix{T}
  x::Vector{T}
  solution::Vector{T}
  energy_solution::T
  initial_energy::T # Used to compute reward, if an improvement is found
  best_found_energy::T
  time::Int  # Count of the steps taken
  history::Vector
end

GI.spec(::GameEnv) = GameSpec()

function GI.init(spec::GameSpec)
  x = random_x(size(spec.Q, 1))
  initial_energy = energy(x, spec.Q) # changed only during reset / clone
  best_found_energy = energy(x, spec.Q)
  time = 0
  history = Vector{Int}()

  # TODO: add to input
  # 1. time since last improvement?
  # 2. ΔE?
  return GameEnv(
    Q=spec.Q,
    x=x,
    solution=spec.solution,
    energy_solution=spec.energy_solution,
    initial_energy=initial_energy,
    best_found_energy=best_found_energy,
    time=time,
    history=history
  )
end

function GI.set_state!(env::GameEnv, state)
  env.x = deepcopy(state.x)
  env.Q = deepcopy(state.Q)
end




function GI.clone(env::GameEnv)
  GameEnv(Q=env.Q, x=deepcopy(env.x), solution=env.solution, energy_solution=env.energy_solution,
    initial_energy=deepcopy(env.initial_energy), best_found_energy=deepcopy(env.best_found_energy), 
    time=deepcopy(env.time), history=deepcopy(env.history))
end

history(::GameEnv) = nothing

#####
##### Defining game rules
#####

#GI.actions_mask(env::GameEnv) = Vector{Bool}([action in env.history ? false : true for action in env.x])

GI.actions_mask(env::GameEnv) = BitVector([1 for _ in env.x])
valid_pos((col, row)) = 1 <= col <= NUM_COLS && 1 <= row <= NUM_ROWS

function GI.play!(env::GameEnv, a)
  env.x = deepcopy(env.x)
  env.x[a] = 1 - env.x[a]
  new_energy = energy(env.x, env.Q)
  env.best_found_energy = min(env.best_found_energy, new_energy)
  push!(env.history, a)

  env.time += 1
end

GI.current_state(env::GameEnv) = begin
  return (Q=env.Q, x=env.x)
end

GI.white_playing(env::GameEnv) = true

#####
##### Reward shaping
#####

function GI.game_terminated(env::GameEnv)
  env.time > EPISODE_LENGTH
end

function GI.white_reward(env::GameEnv)
  if env.time < EPISODE_LENGTH
    return 0.0
  end

  # println("testing for improvement: $(env.best_found_energy) vs $(env.initial_energy)")
  if env.best_found_energy < env.initial_energy
    # println("found improvement: $(env.best_found_energy) < $(env.initial_energy)")
    return reward(env.initial_energy, env.energy_solution, env.best_found_energy)
  else
    return -1
  end
end



#####
##### User interface
#####

function GI.action_string(::GameSpec, a)
  # TODO: why not game env available here?
  return "$a"
end

function GI.parse_action(::GameSpec, s)
  return parse(Int, s)
end

function GI.render(env::GameEnv)
  @show "RL.render"
  println("env.x = $(env.x); energy = $(energy(env.x, env.Q)) env.time = $(env.time); env.best_found_energy = $(env.best_found_energy)")
end

function GI.read_state(::GameSpec)
  nothing
end

GI.heuristic_value(env::GameEnv) = 0.0
