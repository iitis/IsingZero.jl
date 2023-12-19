include("qubo_operations.jl")
using LinearAlgebra
using Statistics: mean

const N = 10
const Q = triu(reshape(Array(1:N^2), (10, 10))) .* repeat([1, -1], 5)
const energy_solution = -1172.0
const solution = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
const episode_length = 7

energy(x, Q) = x' * Q * x
random_x(n) = (abs.(rand(Int, n)) .% 2)
linear_reward(E_initial, E_target, E_current) = ((E_initial - E_current) / (E_initial - E_target))
reward(E_initial, E_target, E_current) = linear_reward(E_initial, E_target, E_current)^2

const AVAILABLE_ACTIONS = collect(1:N)
const baseline_reps = Int(1e6)

# baseline  linear_reward       linear_reward^2     #N
# rand      0.24572             0.10347             1e8
# delta_E   0.97208             0.98441             1e6
# alpha0    0.97                                    4096
# Net Only  ?                   0.44                4096

function random_baseline()
    rewards = Float64[]
    linear_rewards = Float64[]

    for _ in 1:baseline_reps
        x = random_x(N)
        initial_energy = energy(x, Q)
        while initial_energy == energy_solution
            x = random_x(N)
            initial_energy = energy(x, Q)
        end

        best_found_energy = copy(initial_energy)
        for __ in 1:episode_length
            idx = rand(AVAILABLE_ACTIONS)
            @inbounds x[idx] = 1 - x[idx]
            best_found_energy = min(best_found_energy, energy(x, Q))
        end
        
        lr = linear_reward(initial_energy, energy_solution, best_found_energy)
        r = lr ^ 2
        if isnan(r)
            println("?")
        end
        push!(rewards, r)
        push!(linear_rewards, lr)
    end
    return mean(rewards), mean(linear_rewards)
end

function delta_E_baseline()
    rewards = Float64[]
    linear_rewards = Float64[]

    for _ in 1:baseline_reps
        x = random_x(N)
        initial_energy = energy(x, Q)
        while initial_energy == energy_solution
            x = random_x(N)
            initial_energy = energy(x, Q)
        end
        delta_E = compute_delta_E(Q, x)

        best_found_energy = copy(initial_energy)
        for __ in 1:episode_length
            idx = argmin(delta_E)
            update_delta_E!(delta_E, Q, x, idx)
            @inbounds x[idx] = 1 - x[idx]
            best_found_energy = min(best_found_energy, energy(x, Q))
        end
        
        lr = linear_reward(initial_energy, energy_solution, best_found_energy)
        r = lr ^ 2
        if isnan(r)
            println("?")
        end
        push!(rewards, r)
        push!(linear_rewards, lr)
    end
    return mean(rewards), mean(linear_rewards)
end

# println(delta_E_baseline())

using AlphaZero:

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=4096,
    num_workers=512,
    batch_size=64,
    use_gpu=true,
    reset_every=1,# empty MCTS tree every 1 game # TODO???
    flip_probability=0.0,
    alternate_colors=false), mcts=MctsParams(
    num_iters_per_turn=15,
    cpuct=1.0,  # UCT hyperparameter, controls the exploration
    temperature=ConstSchedule(0.0),
    dirichlet_noise_ϵ=0.0,
    dirichlet_noise_α=1.0))
