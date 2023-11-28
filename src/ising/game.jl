using AlphaZero
using CommonRLInterface
using Crayons
using LinearAlgebra
using Base: @kwdef

const RL = CommonRLInterface

const EPISODE_LENGTH = 10

energy(x, Q) = x' * Q * x
random_x(n) = Array{Real}(abs.(rand(Int, n)) .% 2)
reward(E_initial, E_target, E_current) = ((E_initial - E_current) / (E_initial - E_target))

@kwdef mutable struct World <: AbstractEnv
  Q::Array{Real,2}
  x::Array{Real,1}
  solution::Array{Real,1}
  energy_solution::Real
  initial_energy::Real # Used to compute reward, if an improvement is found
  best_found_energy::Real
  time::Int  # Count of the steps taken
end

# TODO: introduce additional structures to organize the World?

function World()
  Q = triu(reshape(Array(1:25), (5, 5))) .* [1.0, -1.0, 1.0, -1.0, 1.0] # Just a temporary, random QUBO
  energy_solution = -64.0
  solution = [0, 1, 0, 1, 1]
  x = random_x(size(Q)[1]) # state
  initial_energy = energy(x, Q) # changed only during reset / clone
  # mutable:
  best_found_energy = energy(x, Q)
  time = 0

  # TODO: add to input
  # 1. time since last improvement?
  # 2. ΔE?
  # return World(Q=Q, x, solution, energy_solution, initial_energy, best_found_energy, time)
  return World(
    Q=Q,
    x=x,
    solution=solution,
    energy_solution=energy_solution,
    initial_energy=initial_energy,
    best_found_energy=best_found_energy,
    time=time
  )
end

RL.reset!(env::World) = begin
  env.x = random_x(size(env.Q)[1])
  env.initial_energy = energy(env.x, env.Q)
  while env.initial_energy == env.energy_solution
    env.x = random_x(size(env.Q)[1])
    env.initial_energy = energy(env.x, env.Q)
  end
  env.best_found_energy = energy(env.x, env.Q)
  env.time = 0
end

RL.actions(env::World) = collect(1:length(env.x))
RL.observe(env::World) = (env.Q, env.x, env.time, env.initial_energy, env.best_found_energy)
RL.terminated(env::World) = env.time > EPISODE_LENGTH

function RL.act!(env::World, a)
  env.x[a] = 1 - env.x[a]
  new_energy = energy(env.x, env.Q)
  env.best_found_energy = min(env.best_found_energy, new_energy)

  env.time += 1

  # What about stacking the reward if the model continues on finding an improvement?
  # How to do it and avoid patologic behaviour?
  if env.time == EPISODE_LENGTH
    @show "testing for improvement: $(env.best_found_energy) vs $(env.initial_energy)"
    if env.best_found_energy < env.initial_energy
      @show "found improvement: $(env.best_found_energy) < $(env.initial_energy)"
      # linear function that would give 0 for no improvement, and 1 for finding the best solution.
      return reward(env.initial_energy, env.energy_solution, env.best_found_energy)
    else
      return -1 # No improvement is punished
    end
  end 
  return 0
end

@provide RL.player(env::World) = 1 # A one player game
@provide RL.players(env::World) = [1]
# @provide RL.observations(env::World) = (env.x, env.time)
@provide RL.clone(env::World) = World(Q=env.Q, x=env.x, solution=env.solution, energy_solution=env.energy_solution, 
initial_energy=env.initial_energy, best_found_energy=env.best_found_energy, time=env.time)
@provide RL.state(env::World) = env.x
# @provide RL.state(env::World) = (env.Q, env.x, env.solution, env.initial_energy, env.best_found_energy)
@provide RL.valid_action_mask(env::World) = begin
  # TODO: This is where we limit flipping of the same index
  return BitVector([1 for _ in env.x])
end
# @provide RL.setstate!(env::World, state) = begin
#   env.Q, env.x, env.solution, env.initial_energy, env.best_found_energy = state
# end
@provide RL.setstate!(env::World, state) = begin
  env.x = state
end

# Additional functions needed by AlphaZero.jl that are not present in 
# CommonRlInterface.jl. Here, we provide them by overriding some functions from
# GameInterface. An alternative would be to pass them as keyword arguments to
# CommonRLInterfaceWrapper.

function GI.render(env::World)
  @show "RL.render"
  println("env.x = $(env.x); env.time = $(env.time); env.best_found_energy = $(env.best_found_energy)")
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
  current_energy = energy(env.x, env.Q)
  new_x = copy(env.x)
  new_x[a] = 1 - new_x[a]
  new_energy = energy(new_x, env.Q)
  return "Flipped idx $a: energy($current_energy ->$new_energy); state($(env.x) -> $new_x)"
end

function GI.parse_action(env::World, s)
  return parse(Int, s)
end

function GI.read_state(env::World)
  return (env.Q, env.x, env.solution, env.initial_energy, env.best_found_energy)
end

# GI.heuristic_value(::World) = 0.0

GameSpec() = CommonRLInterfaceWrapper.Spec(World())