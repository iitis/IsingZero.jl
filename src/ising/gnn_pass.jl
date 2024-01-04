using Flux
using GraphNeuralNetworks



function rand_adj_matrix(n::Int, p::Float64=0.5)
    adj_matrix = zeros(Int, n, n)

    # Populate the upper triangle of the matrix
    for i in 1:n-1
        for j in i+1:n
            if rand() < p
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Symmetry for undirected graph
            end
        end
    end
    adj_matrix
end

N = 10
h1 = 5
h2 = 15
adj_mat = rand_adj_matrix(N)
node_feature_count = 3
node_features_data = rand(node_feature_count, N)
# TODO: consider edge features
# edge_feature_count = 2
# edge_count = sum(adj_mat)
# edge_features_data = rand(edge_features_data, edge_count)


# DATA DEFINITION
input_graph = GNNGraph(adj_mat, ndata=node_features_data)#, edata=edge_features_data)

# MODEL DEFINITION
common = GNNChain(GCNConv(node_feature_count => h1, relu),
                            GCNConv(h1 => h2, relu),
                            GCNConv(h2 => h2, relu),
                            GlobalPool(mean) 
                            )
# pool = GlobalPool(mean) 
vhead = Dense(h2 => 1, tanh)
phead = Dense(h2 => N, sigmoid)


# FORWARD PASS
g = common(input_graph)
# g = pool(g) # attaches gdata.u with shape (hidden_size, node_count)

v = vhead(g.gdata.u)
p = phead(g.gdata.u)
