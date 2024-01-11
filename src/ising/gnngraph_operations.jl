using GraphNeuralNetworks: GNNGraph

function encode_gnngraph(g::GNNGraph)
    # Header: num_nodes, num_edges, and size of node and edge features
    num_nodes = g.num_nodes
    num_edges = g.num_edges
    node_feature_size = size(g.ndata.x, 1)
    edge_feature_size = size(g.edata.e, 1)

    # Flatten node and edge features and convert to Float32
    node_features = Float32.(vec(g.ndata.x))
    edge_features = Float32.(vec(g.edata.e))

    # Process graph structure
    sources, targets, weights = g.graph
    sources = Float32[sources...]
    targets = Float32[targets...]
    weights = Float32[weights...]

    # Concatenate all parts
    encoded_vector = vcat(Float32[num_nodes, num_edges, node_feature_size, edge_feature_size], node_features, sources, targets, weights, edge_features)
    return encoded_vector
end

function decode_gnngraph(encoded_vector)
    # Extract header info
    meta_data_start = 1
    meta_data_end = 4

    num_nodes, num_edges, node_feature_size, edge_feature_size = Int[encoded_vector[meta_data_start:meta_data_end]...]
    
    # Calculate indices for slicing the vector
    node_feature_start = meta_data_end + 1
    node_feature_end = meta_data_end + num_nodes * node_feature_size
    sources_start = node_feature_end + 1
    sources_end = sources_start + num_edges - 1
    targets_start = sources_end + 1
    targets_end = targets_start + num_edges - 1
    weights_start = targets_end + 1
    weights_end = weights_start + num_edges - 1
    edge_features_start = weights_end + 1
    edge_features_end = edge_features_start + num_edges - 1

    # Reconstruct feature matrices
    ndata = reshape(encoded_vector[node_feature_start:node_feature_end], (node_feature_size, num_nodes))
    edata = reshape(encoded_vector[edge_features_start:edge_features_end], (edge_feature_size, num_edges))

    # Reconstruct adjacency matrix
    sources = Int[encoded_vector[sources_start:sources_end]...]
    targets = Int[encoded_vector[targets_start:targets_end]...]
    weights = Int[encoded_vector[weights_start:weights_end]...]

    # Reconstruct GNNGraph
    return GNNGraph((sources, targets, weights), ndata=ndata, edata=edata)
end
