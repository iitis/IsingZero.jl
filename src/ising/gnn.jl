using Graphs
using LinearAlgebra

function qubo_matrix_to_featured_graph(Q, x)
    Q = deepcopy(Q)
    Q = (Q + Q') / 2
    Q[diagind(Q)] .= 0
    A = Q .!= 0
    diag(A) .= 0
    edge_features = Q
    node_features = x'

    return FeaturedGraph(g, ef=edge_features, nf=node_features)
end



