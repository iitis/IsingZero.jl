using LinearAlgebra
using IterTools
using CUDA
using Base.Threads: ReentrantLock

known_Qs = []
Qs_lock = ReentrantLock()

function get_training_Q(::Type{T}, size) where T
    if length(known_Qs) == 0
        lock(Qs_lock)
        try
            if length(known_Qs) == 0
                Q, x = _generate_planted_qubo_CPU(T, size)
                sol_energy = x' * Q * x
                push!(known_Qs, (Q, sol_energy))
                return Q, sol_energy
            end
        finally
            unlock(Qs_lock)
        end
    end
    
    spawn_new_Q = false  #rand() < (1 / 500_000)
    if spawn_new_Q
        @info "Adding new Q to known_Qs, it now has the length of $(length(known_Qs))"
        Q, x = _generate_planted_qubo_CPU(T, size)
        sol_energy = x' * Q * x
        push!(known_Qs, (Q, sol_energy))
        return Q, sol_energy
    else
        return rand(known_Qs)
    end
end


function rand_adj_matrix(n::Int, p::Float64=0.5)
    A = zeros(Int, n, n)

    # Populate the upper triangle of the matrix
    for i in 1:n-1
        for j in i+1:n
            if rand() < p
                A[i, j] = 1
                A[j, i] = 1  # Symmetry for undirected graph
            end
        end
    end
    return sign.(A + A')
end


function _generate_planted_qubo_CPU(::Type{T}, d::Int, n::Int = d) where {T}
    A = rand(T, n, d) .- T(1 / 2)
    x = rand(Bool, d)
    Q = 2 * A' * A
    Q[diagind(Q)] .= T(1 / 2) * Q[diagind(Q)] .- Q * x
    d = diag(Q)
    Q = triu(Q)
    Q[diagind(Q)] = d
    Q, x
end

function _generate_binary_combinations(n)
    comb = collect(product(fill([0, 1], n)...))
    comb = [collect(combo) for combo in comb]
    return comb
end

function _bruteforce_solve(Q)
    comb = _generate_binary_combinations(size(Q)[1])
    comb_cuda = [CuVector{Int}(c) for c in comb]
    comb_cuda = [Vector{Int}(c) for c in comb]
    energies = [x' * Q * x for x in comb_cuda]
    min_energy, idx = findmin(energies)
    solution = comb_cuda[idx]
    return solution, min_energy
end