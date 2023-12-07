using LinearAlgebra

qubo_state(x) = (x + 1) / 2
sigma(x) = 2x - 1
energy(Q, x) = x' * Q * x

function compute_delta_E(Q::Array, x)
    dqx = diag(Q) .* x
    col_sum = vec(sum(Q .* x; dims=1)) .- dqx
    row_sum = vec(sum(Q .* x'; dims=2)) .- dqx
    return -sigma.(x) .* (col_sum .+ row_sum .+ diag(Q))
end

function update_delta_E!(delta_E::Vector, Q::Array, x::Vector, idx::Int)
    new_delta_E_i = -delta_E[idx]
    q_term = Q[idx, :] + Q[:, idx]
    sigma_term = sigma(x[idx]) * sigma.(x)
    delta_E .+= q_term .* sigma_term
    delta_E[idx] = new_delta_E_i
end
