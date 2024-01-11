using Flux
using GraphNeuralNetworks
using Statistics
include("gnngraph_operations.jl")

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

node_count = 10
h1 = 5
h2 = 15
adj_mat = ones(Int, node_count, node_count) #rand_adj_matrix(node_count)
node_feature_count = 3
node_features_data = rand(node_feature_count, node_count)

# TODO: consider edge features
edge_feature_count = 1
edge_count = sum(adj_mat)
edge_features_data = rand(Float32, edge_feature_count, edge_count)


# DATA DEFINITION
input_graph = GNNGraph(adj_mat, ndata=node_features_data, edata=edge_features_data)

input_node_feature_count::Int = 3
edge_feature_dim::Int = 1
hidden_dim1::Int = 320
hidden_dim2::Int = 10


common = GNNChain(
    GATConv((input_node_feature_count, edge_feature_dim) => hidden_dim1, relu, add_self_loops=false),
    GATConv((hidden_dim1, edge_feature_dim) => hidden_dim2, relu, add_self_loops=false),
    GATConv((hidden_dim2, edge_feature_dim) => hidden_dim2, relu, add_self_loops=false),
    GlobalPool(mean)
  )
vhead = Dense(hidden_dim2 => 1, tanh)
# TODO: replace this layer with extra GCNConv and smart pooling to get rid of node_count
phead = Dense(hidden_dim2 => node_count)


# FORWARD PASS
g = common(input_graph)
# g = pool(g) # attaches gdata.u with shape (hidden_size, node_count)

v = vhead(g.gdata.u)
p = phead(g.gdata.u)
println()
println(p, v)

encoded = encode_gnngraph(input_graph)
g2 = decode_gnngraph(encoded)

g = common(g2)
# g = pool(g) # attaches gdata.u with shape (hidden_size, node_count)

v2 = vhead(g.gdata.u)
p2 = phead(g.gdata.u)
println(p2, v2)

