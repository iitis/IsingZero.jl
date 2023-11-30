using AlphaZero
using LinearAlgebra
using Base: @kwdef

struct GameSpec <: GI.AbstractGameSpec
  Q
  energy_solution
  solution

  GameSpec() = begin
    N = 10
    Q = triu(reshape(Array(1:N^2), (N, N))) .* repeat([1, -1], Int(N // 2)) # Just a temporary, random QUBO

    # obrained with bruteforce
    energy_solution = -1172.0
    solution = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1]

    new(Q, energy_solution, solution)
  end

end
const EPISODE_LENGTH = 15
energy(x, Q) = x' * Q * x
random_x(n) = Array{Real}(abs.(rand(Int, n)) .% 2)
reward(E_initial, E_target, E_current) = ((E_initial - E_current) / (E_initial - E_target))

@kwdef mutable struct GameEnv <: GI.AbstractGameEnv
  Q::Array{Real,2}
  x::Array{Real,1}
  solution::Array{Real,1}
  energy_solution::Real
  initial_energy::Real # Used to compute reward, if an improvement is found
  best_found_energy::Real
  time::Int  # Count of the steps taken
end

GI.spec(::GameEnv) = GameSpec()

function GI.init(spec::GameSpec)
  x = random_x(size(spec.Q)[1])
  initial_energy = energy(x, spec.Q) # changed only during reset / clone
  best_found_energy = energy(x, spec.Q)
  time = 0

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
    time=time
  )
end

function GI.set_state!(env::GameEnv, state)
  env.x = deepcopy(state.x)
  env.Q = deepcopy(state.Q)
end

GI.two_players(::GameSpec) = false

GI.actions(spec::GameSpec) = [i for (i, _) in enumerate(spec.solution)]

function GI.clone(env::GameEnv)
  GameEnv(Q=env.Q, x=deepcopy(env.x), solution=env.solution, energy_solution=env.energy_solution,
    initial_energy=deepcopy(env.initial_energy), best_found_energy=deepcopy(env.best_found_energy), 
    time=deepcopy(env.time))
end

history(env::GameEnv) = nothing

#####
##### Defining game rules
#####

GI.actions_mask(env::GameEnv) = BitVector([1 for _ in env.x])

valid_pos((col, row)) = 1 <= col <= NUM_COLS && 1 <= row <= NUM_ROWS

function GI.play!(env::GameEnv, a)
  env.x = deepcopy(env.x)
  env.x[a] = 1 - env.x[a]
  new_energy = energy(env.x, env.Q)
  env.best_found_energy = min(env.best_found_energy, new_energy)

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
##### User interface
#####

function GI.action_string(spec::GameSpec, a)
  # TODO: why not game env available here?
  return "Flipped idx $a"
end

function GI.parse_action(spec::GameSpec, s)
  return parse(Int, s)
end

function GI.render(env::GameEnv)
  @show "RL.render"
  println("env.x = $(env.x); env.time = $(env.time); env.best_found_energy = $(env.best_found_energy)")
end

function GI.read_state(spec::GameSpec)
  nothing
end

GI.heuristic_value(env::GameEnv) = 0.0
