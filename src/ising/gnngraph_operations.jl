using GraphNeuralNetworks: GNNGraph

function encode_gnngraph(g::GNNGraph)
    # Header: num_nodes, num_edges, and size of node features
    num_nodes = g.num_nodes
    num_edges = g.num_edges
    feature_size = size(g.ndata.x, 1)  # Assuming ndata.x is the node feature matrix

    # Flatten node features and convert to Float32
    node_features = Float32[reshape(g.ndata.x, :)...]

    # Process graph structure
    sources, targets, weights = g.graph  # Assuming COO format with optional weights
    sources = Float32[sources...]
    targets = Float32[targets...]
    weights = Float32[weights...]

    # Concatenate all parts
    encoded_vector = vcat(Float32[num_nodes, num_edges, feature_size], node_features, sources, targets, weights)
    return encoded_vector
end

function decode_gnngraph(encoded_vector)
    # Extract header info
    meta_data_start = 1
    meta_data_end = 3

    num_nodes, num_edges, feature_size = Int[encoded_vector[meta_data_start:meta_data_end]...]

    # Calculate indices for slicing the vector
    node_feature_end = meta_data_end + num_nodes * feature_size
    sources_start = node_feature_end + 1
    sources_end = sources_start + num_edges - 1
    targets_start = sources_end + 1
    targets_end = targets_start + num_edges - 1
    weights_start = targets_end + 1
    weights_end = weights_start + num_edges - 1

    # Reconstruct node features
    ndata = reshape(encoded_vector[4:node_feature_end], feature_size, num_nodes)

    # Reconstruct adjacency matrix
    sources = Int[encoded_vector[sources_start:sources_end]...]
    targets = Int[encoded_vector[targets_start:targets_end]...]
    weights = Int[encoded_vector[weights_start:weights_end]...]

    # Reconstruct GNNGraph
    return GNNGraph((sources, targets, weights), ndata=ndata)
end
