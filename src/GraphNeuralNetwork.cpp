//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Spatiotemporal & Graph Neural Networks Implementation
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 2025
*****************************************************************************/

#include "GraphNeuralNetwork.h"
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>

namespace ML {
namespace GNN {

// GraphStructure Implementation
size_t GraphStructure::add_node(const std::vector<float>& features, const std::string& type) {
    assert(features.size() == feature_dim && "Feature dimension mismatch");
    
    size_t node_id = nodes.size();
    nodes.emplace_back(node_id, features, type);
    
    if (!type.empty()) {
        node_type_mapping[type].push_back(node_id);
    }
    
    adjacency_list[node_id] = std::vector<size_t>();
    return node_id;
}

void GraphStructure::add_edge(size_t source, size_t target, float weight, const std::string& type) {
    assert(source < nodes.size() && "Source node does not exist");
    assert(target < nodes.size() && "Target node does not exist");
    
    edges.emplace_back(source, target, weight, type);
    
    adjacency_list[source].push_back(target);
    if (!is_directed) {
        adjacency_list[target].push_back(source);
    }
    
    if (!type.empty()) {
        edge_type_mapping[type].push_back(edges.size() - 1);
    }
}

void GraphStructure::remove_node(size_t node_id) {
    if (node_id >= nodes.size()) return;
    
    // Remove all edges connected to this node
    edges.erase(std::remove_if(edges.begin(), edges.end(),
                [node_id](const Edge& e) {
                    return e.source == node_id || e.target == node_id;
                }), edges.end());
    
    // Remove from adjacency lists
    for (auto& pair : adjacency_list) {
        auto& neighbors = pair.second;
        neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), node_id), neighbors.end());
    }
    adjacency_list.erase(node_id);
    
    // Remove node
    nodes.erase(nodes.begin() + node_id);
    
    // Update node IDs and mappings
    for (size_t i = node_id; i < nodes.size(); ++i) {
        nodes[i].id = i;
    }
    
    for (auto& pair : node_type_mapping) {
        auto& node_list = pair.second;
        std::replace(node_list.begin(), node_list.end(), node_id, SIZE_MAX);
        node_list.erase(std::remove(node_list.begin(), node_list.end(), SIZE_MAX), node_list.end());
        for (auto& id : node_list) {
            if (id > node_id) id--;
        }
    }
}

void GraphStructure::remove_edge(size_t source, size_t target) {
    edges.erase(std::remove_if(edges.begin(), edges.end(),
                [this, source, target](const Edge& e) {
                    return (e.source == source && e.target == target) ||
                           (!is_directed && e.source == target && e.target == source);
                }), edges.end());
    
    auto& neighbors = adjacency_list[source];
    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), target), neighbors.end());
    
    if (!is_directed) {
        auto& reverse_neighbors = adjacency_list[target];
        reverse_neighbors.erase(std::remove(reverse_neighbors.begin(), reverse_neighbors.end(), source), 
                               reverse_neighbors.end());
    }
}

const Node& GraphStructure::get_node(size_t node_id) const {
    assert(node_id < nodes.size() && "Node does not exist");
    return nodes[node_id];
}

const Edge& GraphStructure::get_edge(size_t edge_id) const {
    assert(edge_id < edges.size() && "Edge does not exist");
    return edges[edge_id];
}

const std::vector<size_t>& GraphStructure::get_neighbors(size_t node_id) const {
    static const std::vector<size_t> empty_neighbors;
    auto it = adjacency_list.find(node_id);
    return (it != adjacency_list.end()) ? it->second : empty_neighbors;
}

const std::vector<size_t>& GraphStructure::get_nodes_by_type(const std::string& type) const {
    static const std::vector<size_t> empty_nodes;
    auto it = node_type_mapping.find(type);
    return (it != node_type_mapping.end()) ? it->second : empty_nodes;
}

const std::vector<size_t>& GraphStructure::get_edges_by_type(const std::string& type) const {
    static const std::vector<size_t> empty_edges;
    auto it = edge_type_mapping.find(type);
    return (it != edge_type_mapping.end()) ? it->second : empty_edges;
}

std::vector<std::string> GraphStructure::get_node_types() const {
    std::vector<std::string> types;
    for (const auto& pair : node_type_mapping) {
        types.push_back(pair.first);
    }
    return types;
}

std::vector<std::string> GraphStructure::get_edge_types() const {
    std::vector<std::string> types;
    for (const auto& pair : edge_type_mapping) {
        types.push_back(pair.first);
    }
    return types;
}

void GraphStructure::normalize_features() {
    if (nodes.empty()) return;
    
    // Compute mean and std dev for each feature dimension
    std::vector<float> mean(feature_dim, 0.0f);
    std::vector<float> std_dev(feature_dim, 0.0f);
    
    for (const auto& node : nodes) {
        for (size_t i = 0; i < feature_dim; ++i) {
            mean[i] += node.features[i];
        }
    }
    
    for (size_t i = 0; i < feature_dim; ++i) {
        mean[i] /= nodes.size();
    }
    
    for (const auto& node : nodes) {
        for (size_t i = 0; i < feature_dim; ++i) {
            float diff = node.features[i] - mean[i];
            std_dev[i] += diff * diff;
        }
    }
    
    for (size_t i = 0; i < feature_dim; ++i) {
        std_dev[i] = std::sqrt(std_dev[i] / nodes.size());
        if (std_dev[i] < 1e-8f) std_dev[i] = 1.0f;  // Avoid division by zero
    }
    
    // Normalize features
    for (auto& node : nodes) {
        for (size_t i = 0; i < feature_dim; ++i) {
            node.features[i] = (node.features[i] - mean[i]) / std_dev[i];
        }
    }
}

void GraphStructure::compute_laplacian(std::vector<float>& laplacian) const {
    size_t n = nodes.size();
    laplacian.assign(n * n, 0.0f);
    
    std::vector<float> degree = get_degree_matrix();
    std::vector<float> adjacency = get_adjacency_matrix();
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            laplacian[i * n + j] = degree[i * n + j] - adjacency[i * n + j];
        }
    }
}

std::vector<float> GraphStructure::get_degree_matrix() const {
    size_t n = nodes.size();
    std::vector<float> degree(n * n, 0.0f);
    
    for (size_t i = 0; i < n; ++i) {
        auto it = adjacency_list.find(i);
        if (it != adjacency_list.end()) {
            degree[i * n + i] = static_cast<float>(it->second.size());
        }
    }
    
    return degree;
}

std::vector<float> GraphStructure::get_adjacency_matrix() const {
    size_t n = nodes.size();
    std::vector<float> adjacency(n * n, 0.0f);
    
    for (const auto& edge : edges) {
        adjacency[edge.source * n + edge.target] = edge.weight;
        if (!is_directed) {
            adjacency[edge.target * n + edge.source] = edge.weight;
        }
    }
    
    return adjacency;
}

void GraphStructure::clear() {
    nodes.clear();
    edges.clear();
    adjacency_list.clear();
    node_type_mapping.clear();
    edge_type_mapping.clear();
}

// MessagePassingLayer Implementation
MessagePassingLayer::MessagePassingLayer(size_t in_dim, size_t out_dim, 
                                         float dropout, bool batch_norm)
    : input_dim(in_dim), output_dim(out_dim), dropout_rate(dropout), 
      use_batch_norm(batch_norm) {
    
    message_weights.assign(in_dim * out_dim, 0.0f);
    update_weights.assign(out_dim * out_dim, 0.0f);
    message_bias.assign(out_dim, 0.0f);
    update_bias.assign(out_dim, 0.0f);
    
    // Initialize weights with Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f / std::sqrt(in_dim), 1.0f / std::sqrt(in_dim));
    std::uniform_real_distribution<float> update_dis(-1.0f / std::sqrt(out_dim), 1.0f / std::sqrt(out_dim));
    
    for (auto& w : message_weights) {
        w = dis(gen);
    }
    for (auto& w : update_weights) {
        w = update_dis(gen);
    }
}

void MessagePassingLayer::forward(const GraphStructure& graph, 
                                 const std::vector<float>& input_features,
                                 std::vector<float>& output_features) {
    size_t num_nodes = graph.num_nodes();
    std::vector<std::vector<float>> messages(num_nodes);
    
    // Message phase
    message_function(graph, input_features, messages);
    
    // Aggregate phase
    std::vector<float> aggregated(num_nodes * output_dim, 0.0f);
    aggregate_function(graph, messages, aggregated);
    
    // Update phase
    update_function(aggregated, output_features);
}

void MessagePassingLayer::apply_activation(std::vector<float>& x) const {
    // ReLU activation
    std::transform(x.begin(), x.end(), x.begin(),
                   [](float v) { return std::max(0.0f, v); });
}

void MessagePassingLayer::apply_batch_norm(std::vector<float>& x) const {
    if (!use_batch_norm) return;
    
    size_t num_elements = x.size();
    if (num_elements == 0) return;
    
    // Compute mean
    float mean = std::accumulate(x.begin(), x.end(), 0.0f) / num_elements;
    
    // Compute variance
    float variance = 0.0f;
    for (float v : x) {
        variance += (v - mean) * (v - mean);
    }
    variance /= num_elements;
    
    // Normalize
    float std_dev = std::sqrt(variance + 1e-8f);
    for (auto& v : x) {
        v = (v - mean) / std_dev;
    }
}

void MessagePassingLayer::apply_dropout(std::vector<float>& x, std::mt19937& rng) const {
    if (dropout_rate <= 0.0f) return;
    
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto& v : x) {
        if (dis(rng) < dropout_rate) {
            v = 0.0f;
        } else {
            v /= (1.0f - dropout_rate);  // Inverted dropout
        }
    }
}

// GraphConvolutionalLayer Implementation
GraphConvolutionalLayer::GraphConvolutionalLayer(size_t in_dim, size_t out_dim, 
                                                 bool normalize_adj, bool self_loops,
                                                 float dropout, bool batch_norm)
    : MessagePassingLayer(in_dim, out_dim, dropout, batch_norm),
      normalize_adjacency(normalize_adj), use_self_loops(self_loops) {}

void GraphConvolutionalLayer::message_function(const GraphStructure& graph, 
                                             const std::vector<float>& node_features,
                                             std::vector<std::vector<float>>& messages) const {
    size_t num_nodes = graph.num_nodes();
    
    for (size_t i = 0; i < num_nodes; ++i) {
        messages[i].assign(output_dim, 0.0f);
        
        // Transform node features
        for (size_t j = 0; j < input_dim; ++j) {
            for (size_t k = 0; k < output_dim; ++k) {
                messages[i][k] += node_features[i * input_dim + j] * 
                                 message_weights[j * output_dim + k];
            }
        }
        
        // Add bias
        for (size_t k = 0; k < output_dim; ++k) {
            messages[i][k] += message_bias[k];
        }
    }
}

void GraphConvolutionalLayer::aggregate_function(const GraphStructure& graph,
                                                const std::vector<std::vector<float>>& messages,
                                                std::vector<float>& aggregated) const {
    size_t num_nodes = graph.num_nodes();
    std::fill(aggregated.begin(), aggregated.end(), 0.0f);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& neighbors = graph.get_neighbors(i);
        
        // Include self if using self-loops
        std::vector<size_t> agg_neighbors = neighbors;
        if (use_self_loops) {
            agg_neighbors.push_back(i);
        }
        
        if (agg_neighbors.empty()) continue;
        
        // Aggregate messages from neighbors
        for (size_t neighbor : agg_neighbors) {
            for (size_t k = 0; k < output_dim; ++k) {
                aggregated[i * output_dim + k] += messages[neighbor][k];
            }
        }
        
        // Normalize by degree if requested
        if (normalize_adjacency) {
            float norm_factor = 1.0f / std::sqrt(agg_neighbors.size());
            for (size_t k = 0; k < output_dim; ++k) {
                aggregated[i * output_dim + k] *= norm_factor;
            }
        }
    }
}

void GraphConvolutionalLayer::update_function(const std::vector<float>& aggregated,
                                            std::vector<float>& output) const {
    size_t num_nodes = aggregated.size() / output_dim;
    output.assign(num_nodes * output_dim, 0.0f);
    
    // Apply linear transformation
    for (size_t i = 0; i < num_nodes; ++i) {
        for (size_t j = 0; j < output_dim; ++j) {
            for (size_t k = 0; k < output_dim; ++k) {
                output[i * output_dim + k] += aggregated[i * output_dim + j] *
                                              update_weights[j * output_dim + k];
            }
        }
        
        // Add bias
        for (size_t k = 0; k < output_dim; ++k) {
            output[i * output_dim + k] += update_bias[k];
        }
        
        // Apply activation
        std::vector<float> node_output(output.begin() + i * output_dim, 
                                     output.begin() + (i + 1) * output_dim);
        apply_activation(node_output);
        std::copy(node_output.begin(), node_output.end(), 
                 output.begin() + i * output_dim);
    }
}

// GraphAttentionLayer Implementation
GraphAttentionLayer::GraphAttentionLayer(size_t in_dim, size_t out_dim, size_t heads,
                                         float neg_slope, bool residual,
                                         float dropout, bool batch_norm)
    : MessagePassingLayer(in_dim, out_dim, dropout, batch_norm),
      num_heads(heads), negative_slope(neg_slope), use_residual(residual) {
    
    attention_weights.assign(out_dim * 2, 0.0f);  // For source and target attention
    
    // Initialize attention weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f / std::sqrt(out_dim), 1.0f / std::sqrt(out_dim));
    
    for (auto& w : attention_weights) {
        w = dis(gen);
    }
}

void GraphAttentionLayer::message_function(const GraphStructure& graph, 
                                         const std::vector<float>& node_features,
                                         std::vector<std::vector<float>>& messages) const {
    size_t num_nodes = graph.num_nodes();
    
    for (size_t i = 0; i < num_nodes; ++i) {
        messages[i].assign(output_dim, 0.0f);
        
        // Transform node features
        for (size_t j = 0; j < input_dim; ++j) {
            for (size_t k = 0; k < output_dim; ++k) {
                messages[i][k] += node_features[i * input_dim + j] * 
                                 message_weights[j * output_dim + k];
            }
        }
        
        // Add bias
        for (size_t k = 0; k < output_dim; ++k) {
            messages[i][k] += message_bias[k];
        }
    }
}

void GraphAttentionLayer::aggregate_function(const GraphStructure& graph,
                                            const std::vector<std::vector<float>>& messages,
                                            std::vector<float>& aggregated) const {
    size_t num_nodes = graph.num_nodes();
    std::fill(aggregated.begin(), aggregated.end(), 0.0f);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& neighbors = graph.get_neighbors(i);
        
        if (neighbors.empty()) {
            // If no neighbors, use self-message
            for (size_t k = 0; k < output_dim; ++k) {
                aggregated[i * output_dim + k] = messages[i][k];
            }
            continue;
        }
        
        // Compute attention scores
        std::vector<float> attention_scores;
        for (size_t neighbor : neighbors) {
            std::vector<float> scores = compute_attention_scores(
                std::vector<float>(messages[i].begin(), messages[i].end()),
                std::vector<float>(messages[neighbor].begin(), messages[neighbor].end())
            );
            attention_scores.insert(attention_scores.end(), scores.begin(), scores.end());
        }
        
        // Apply softmax to attention scores
        float sum_exp = 0.0f;
        for (float score : attention_scores) {
            sum_exp += std::exp(score);
        }
        
        // Aggregate messages with attention weights
        for (size_t idx = 0; idx < neighbors.size(); ++idx) {
            size_t neighbor = neighbors[idx];
            float attention_weight = std::exp(attention_scores[idx]) / sum_exp;
            
            for (size_t k = 0; k < output_dim; ++k) {
                aggregated[i * output_dim + k] += attention_weight * messages[neighbor][k];
            }
        }
    }
}

void GraphAttentionLayer::update_function(const std::vector<float>& aggregated,
                                         std::vector<float>& output) const {
    size_t num_nodes = aggregated.size() / output_dim;
    output.assign(num_nodes * output_dim, 0.0f);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        for (size_t j = 0; j < output_dim; ++j) {
            for (size_t k = 0; k < output_dim; ++k) {
                output[i * output_dim + k] += aggregated[i * output_dim + j] *
                                              update_weights[j * output_dim + k];
            }
        }
        
        // Add bias
        for (size_t k = 0; k < output_dim; ++k) {
            output[i * output_dim + k] += update_bias[k];
        }
        
        // Apply activation
        std::vector<float> node_output(output.begin() + i * output_dim, 
                                     output.begin() + (i + 1) * output_dim);
        apply_activation(node_output);
        std::copy(node_output.begin(), node_output.end(), 
                 output.begin() + i * output_dim);
        
        // Add residual connection if enabled
        if (use_residual) {
            for (size_t k = 0; k < output_dim; ++k) {
                output[i * output_dim + k] += aggregated[i * output_dim + k];
            }
        }
    }
}

std::vector<float> GraphAttentionLayer::compute_attention_scores(const std::vector<float>& source_feat,
                                                                const std::vector<float>& target_feat) const {
    std::vector<float> scores(num_heads, 0.0f);
    
    // Concatenate source and target features
    std::vector<float> combined;
    combined.insert(combined.end(), source_feat.begin(), source_feat.end());
    combined.insert(combined.end(), target_feat.begin(), target_feat.end());
    
    // Compute attention scores for each head
    for (size_t head = 0; head < num_heads; ++head) {
        for (size_t i = 0; i < combined.size(); ++i) {
            scores[head] += combined[i] * attention_weights[i];
        }
        
        // Apply LeakyReLU
        scores[head] = scores[head] >= 0 ? scores[head] : negative_slope * scores[head];
    }
    
    return scores;
}

// GraphSAGELayer Implementation
GraphSAGELayer::GraphSAGELayer(size_t in_dim, size_t out_dim, 
                               const std::string& agg_type, size_t samples, bool normalize,
                               float dropout, bool batch_norm)
    : MessagePassingLayer(in_dim, out_dim, dropout, batch_norm),
      aggregator_type(agg_type), num_samples(samples), normalize_features(normalize) {}

void GraphSAGELayer::message_function(const GraphStructure& graph, 
                                     const std::vector<float>& node_features,
                                     std::vector<std::vector<float>>& messages) const {
    size_t num_nodes = graph.num_nodes();
    
    for (size_t i = 0; i < num_nodes; ++i) {
        messages[i].assign(output_dim, 0.0f);

        // Linear transform of node features into message space.
        for (size_t j = 0; j < input_dim; ++j) {
            for (size_t k = 0; k < output_dim; ++k) {
                messages[i][k] += node_features[i * input_dim + j] *
                                  message_weights[j * output_dim + k];
            }
        }

        for (size_t k = 0; k < output_dim; ++k) {
            messages[i][k] += message_bias[k];
        }
    }
}

void GraphSAGELayer::aggregate_function(const GraphStructure& graph,
                                       const std::vector<std::vector<float>>& messages,
                                       std::vector<float>& aggregated) const {
    size_t num_nodes = graph.num_nodes();
    
    for (size_t i = 0; i < num_nodes; ++i) {
        for (size_t k = 0; k < output_dim; ++k) {
            aggregated[i * output_dim + k] = messages[i][k];
        }
        
        if (normalize_features) {
            // L2 normalization
            float norm = 0.0f;
            for (size_t k = 0; k < output_dim; ++k) {
                norm += aggregated[i * output_dim + k] * aggregated[i * output_dim + k];
            }
            norm = std::sqrt(norm + 1e-8f);
            
            for (size_t k = 0; k < output_dim; ++k) {
                aggregated[i * output_dim + k] /= norm;
            }
        }
    }
}

void GraphSAGELayer::update_function(const std::vector<float>& aggregated,
                                   std::vector<float>& output) const {
    size_t num_nodes = aggregated.size() / output_dim;
    output.assign(num_nodes * output_dim, 0.0f);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        for (size_t j = 0; j < output_dim; ++j) {
            for (size_t k = 0; k < output_dim; ++k) {
                output[i * output_dim + k] += aggregated[i * output_dim + j] *
                                              update_weights[j * output_dim + k];
            }
        }
        
        // Add bias
        for (size_t k = 0; k < output_dim; ++k) {
            output[i * output_dim + k] += update_bias[k];
        }
        
        // Apply activation
        std::vector<float> node_output(output.begin() + i * output_dim, 
                                     output.begin() + (i + 1) * output_dim);
        apply_activation(node_output);
        std::copy(node_output.begin(), node_output.end(), 
                 output.begin() + i * output_dim);
    }
}

void GraphSAGELayer::mean_aggregation(const std::vector<std::vector<float>>& messages,
                                      std::vector<float>& aggregated) const {
    // Implementation would go here - simplified for brevity
}

void GraphSAGELayer::max_aggregation(const std::vector<std::vector<float>>& messages,
                                     std::vector<float>& aggregated) const {
    // Implementation would go here - simplified for brevity
}

void GraphSAGELayer::lstm_aggregation(const std::vector<std::vector<float>>& messages,
                                      std::vector<float>& aggregated) const {
    // Implementation would go here - simplified for brevity
}

// TemporalGraphLayer Implementation
TemporalGraphLayer::TemporalGraphLayer(size_t in_dim, size_t out_dim, size_t t_steps,
                                      float decay, bool gru, float dropout, bool batch_norm)
    : MessagePassingLayer(in_dim, out_dim, dropout, batch_norm),
      time_steps(t_steps), time_decay_factor(decay), use_gru_update(gru) {
    
    temporal_weights.assign(in_dim * out_dim, 0.0f);
    
    // Initialize temporal weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f / std::sqrt(in_dim), 1.0f / std::sqrt(in_dim));
    
    for (auto& w : temporal_weights) {
        w = dis(gen);
    }
}

void TemporalGraphLayer::message_function(const GraphStructure& graph, 
                                         const std::vector<float>& node_features,
                                         std::vector<std::vector<float>>& messages) const {
    size_t num_nodes = graph.num_nodes();
    
    for (size_t i = 0; i < num_nodes; ++i) {
        messages[i].assign(output_dim, 0.0f);
        
        // Transform node features with temporal weights
        for (size_t j = 0; j < input_dim; ++j) {
            for (size_t k = 0; k < output_dim; ++k) {
                messages[i][k] += node_features[i * input_dim + j] * 
                                 temporal_weights[j * output_dim + k];
            }
        }
        
        // Add bias
        for (size_t k = 0; k < output_dim; ++k) {
            messages[i][k] += message_bias[k];
        }
    }
}

void TemporalGraphLayer::aggregate_function(const GraphStructure& graph,
                                           const std::vector<std::vector<float>>& messages,
                                           std::vector<float>& aggregated) const {
    size_t num_nodes = graph.num_nodes();
    std::fill(aggregated.begin(), aggregated.end(), 0.0f);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& neighbors = graph.get_neighbors(i);
        
        if (neighbors.empty()) {
            for (size_t k = 0; k < output_dim; ++k) {
                aggregated[i * output_dim + k] = messages[i][k];
            }
            continue;
        }
        
        // Aggregate with temporal decay
        for (size_t neighbor : neighbors) {
            for (size_t k = 0; k < output_dim; ++k) {
                aggregated[i * output_dim + k] += messages[neighbor][k] * time_decay_factor;
            }
        }
        
        // Normalize by number of neighbors
        float norm_factor = 1.0f / neighbors.size();
        for (size_t k = 0; k < output_dim; ++k) {
            aggregated[i * output_dim + k] *= norm_factor;
        }
    }
}

void TemporalGraphLayer::update_function(const std::vector<float>& aggregated,
                                        std::vector<float>& output) const {
    size_t num_nodes = aggregated.size() / output_dim;
    output.assign(num_nodes * output_dim, 0.0f);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        if (use_gru_update) {
            // GRU-style update
            std::vector<float> input(aggregated.begin() + i * output_dim,
                                     aggregated.begin() + (i + 1) * output_dim);
            std::vector<float> hidden(output.begin() + i * output_dim,
                                      output.begin() + (i + 1) * output_dim);
            std::vector<float> updated(output_dim, 0.0f);
            gru_update(input, hidden, updated);
            std::copy(updated.begin(), updated.end(), output.begin() + i * output_dim);
        } else {
            // Standard update
            for (size_t j = 0; j < output_dim; ++j) {
                for (size_t k = 0; k < output_dim; ++k) {
                    output[i * output_dim + k] += aggregated[i * output_dim + j] *
                                                  update_weights[j * output_dim + k];
                }
            }
            
            // Add bias
            for (size_t k = 0; k < output_dim; ++k) {
                output[i * output_dim + k] += update_bias[k];
            }
        }
        
        // Apply activation
        std::vector<float> node_output(output.begin() + i * output_dim, 
                                     output.begin() + (i + 1) * output_dim);
        apply_activation(node_output);
        std::copy(node_output.begin(), node_output.end(), 
                 output.begin() + i * output_dim);
    }
}

void TemporalGraphLayer::forward_temporal(const std::vector<GraphStructure>& graph_sequence,
                                         const std::vector<std::vector<float>>& feature_sequence,
                                         std::vector<float>& output_features) {
    if (graph_sequence.empty() || feature_sequence.empty()) return;
    
    size_t current_time = 0;
    std::vector<float> current_features = feature_sequence[0];
    
    for (size_t t = 0; t < time_steps && t < graph_sequence.size(); ++t) {
        // Apply temporal decay
        apply_temporal_decay(current_features, t);
        
        // Forward pass through current graph
        std::vector<float> temp_output;
        forward(graph_sequence[t], current_features, temp_output);
        
        current_features = temp_output;
        current_time = t;
    }
    
    output_features = current_features;
}

void TemporalGraphLayer::apply_temporal_decay(std::vector<float>& features, size_t time_step) const {
    float decay = std::pow(time_decay_factor, static_cast<float>(time_step));
    for (auto& f : features) {
        f *= decay;
    }
}

void TemporalGraphLayer::gru_update(const std::vector<float>& input, const std::vector<float>& hidden,
                                   std::vector<float>& output) const {
    // Simplified GRU update
    for (size_t i = 0; i < output_dim; ++i) {
        float z = 1.0f / (1.0f + std::exp(-(input[i] + hidden[i])));  // Update gate
        float r = 1.0f / (1.0f + std::exp(-(input[i] - hidden[i])));  // Reset gate
        output[i] = (1.0f - z) * hidden[i] + z * std::tanh(input[i] + r * hidden[i]);
    }
}

// SpatiotemporalTransformer Implementation
SpatiotemporalTransformer::SpatiotemporalTransformer(size_t spatial_d, size_t temporal_d, 
                                                    size_t heads, size_t layers, float dropout)
    : spatial_dim(spatial_d), temporal_dim(temporal_d), num_heads(heads), 
      num_layers(layers), dropout_rate(dropout) {
    
    spatial_attention_weights.assign(spatial_dim * spatial_dim * heads, 0.0f);
    temporal_attention_weights.assign(temporal_dim * temporal_dim * heads, 0.0f);
    fusion_weights.assign(spatial_dim + temporal_dim, 0.0f);
    
    // Initialize weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f / std::sqrt(spatial_dim + temporal_dim), 
                                             1.0f / std::sqrt(spatial_dim + temporal_dim));
    
    for (auto& w : spatial_attention_weights) {
        w = dis(gen);
    }
    for (auto& w : temporal_attention_weights) {
        w = dis(gen);
    }
    for (auto& w : fusion_weights) {
        w = dis(gen);
    }
}

void SpatiotemporalTransformer::forward(const GraphStructure& graph,
                                       const std::vector<std::vector<float>>& temporal_features,
                                       std::vector<float>& output) {
    std::vector<float> spatial_output(spatial_dim);
    std::vector<float> temporal_output(temporal_dim);
    
    // Spatial attention
    if (!graph.empty()) {
        std::vector<float> node_features;
        for (const auto& node : graph.get_nodes()) {
            node_features.insert(node_features.end(), node.features.begin(), node.features.end());
        }
        spatial_attention(graph, node_features, spatial_output);
    }
    
    // Temporal attention
    if (!temporal_features.empty()) {
        temporal_attention(temporal_features, temporal_output);
    }
    
    // Space-time fusion
    spatiotemporal_fusion(spatial_output, temporal_output, output);
}

void SpatiotemporalTransformer::spatial_attention(const GraphStructure& graph,
                                                  const std::vector<float>& features,
                                                  std::vector<float>& spatial_output) {
    compute_spatial_attention_weights(graph, features);
    
    spatial_output.assign(spatial_dim, 0.0f);
    
    // Apply multi-head attention
    for (size_t head = 0; head < num_heads; ++head) {
        for (size_t i = 0; i < spatial_dim; ++i) {
            for (size_t j = 0; j < spatial_dim; ++j) {
                spatial_output[i] += features[j] * 
                    spatial_attention_weights[(head * spatial_dim + i) * spatial_dim + j];
            }
        }
    }
    
    // Normalize by number of heads
    for (auto& val : spatial_output) {
        val /= num_heads;
    }
}

void SpatiotemporalTransformer::temporal_attention(const std::vector<std::vector<float>>& temporal_seq,
                                                  std::vector<float>& temporal_output) {
    compute_temporal_attention_weights(temporal_seq);
    
    temporal_output.assign(temporal_dim, 0.0f);
    
    // Apply temporal attention
    for (size_t head = 0; head < num_heads; ++head) {
        for (size_t t = 0; t < temporal_seq.size() && t < temporal_dim; ++t) {
            for (size_t i = 0; i < temporal_dim; ++i) {
                if (i < temporal_seq[t].size()) {
                    temporal_output[i] += temporal_seq[t][i] * 
                        temporal_attention_weights[(head * temporal_dim + t) * temporal_dim + i];
                }
            }
        }
    }
    
    // Normalize by number of heads
    for (auto& val : temporal_output) {
        val /= num_heads;
    }
}

void SpatiotemporalTransformer::spatiotemporal_fusion(const std::vector<float>& spatial_feat,
                                                     const std::vector<float>& temporal_feat,
                                                     std::vector<float>& fused_output) {
    size_t fused_dim = spatial_dim + temporal_dim;
    fused_output.assign(fused_dim, 0.0f);
    
    // Concatenate spatial and temporal features
    std::vector<float> combined = spatial_feat;
    combined.insert(combined.end(), temporal_feat.begin(), temporal_feat.end());
    
    // Apply fusion weights
    for (size_t i = 0; i < fused_dim; ++i) {
        fused_output[i] = combined[i] * fusion_weights[i];
    }
}

void SpatiotemporalTransformer::compute_spatial_attention_weights(const GraphStructure& graph,
                                                                 const std::vector<float>& features) {
    // Simplified spatial attention computation
    for (size_t head = 0; head < num_heads; ++head) {
        for (size_t i = 0; i < spatial_dim; ++i) {
            for (size_t j = 0; j < spatial_dim; ++j) {
                float score = 0.0f;
                if (i < features.size() && j < features.size()) {
                    score = features[i] * features[j];
                }
                spatial_attention_weights[(head * spatial_dim + i) * spatial_dim + j] = score;
            }
        }
    }
}

void SpatiotemporalTransformer::compute_temporal_attention_weights(const std::vector<std::vector<float>>& temporal_seq) {
    // Simplified temporal attention computation
    for (size_t head = 0; head < num_heads; ++head) {
        for (size_t t = 0; t < temporal_seq.size() && t < temporal_dim; ++t) {
            for (size_t i = 0; i < temporal_dim; ++i) {
                float score = 0.0f;
                if (t < temporal_seq.size() && i < temporal_seq[t].size()) {
                    score = temporal_seq[t][i] * temporal_seq[t][i];
                }
                temporal_attention_weights[(head * temporal_dim + t) * temporal_dim + i] = score;
            }
        }
    }
}

// GraphNeuralODE Implementation
GraphNeuralODE::GraphNeuralODE(size_t feat_dim, float int_time, const std::string& method)
    : feature_dim(feat_dim), integration_time(int_time), integration_method(method) {
    
    ode_weights.assign(feat_dim * feat_dim, 0.0f);
    
    // Initialize ODE weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f / std::sqrt(feat_dim), 1.0f / std::sqrt(feat_dim));
    
    for (auto& w : ode_weights) {
        w = dis(gen);
    }
}

void GraphNeuralODE::forward(const GraphStructure& graph,
                            const std::vector<float>& initial_features,
                            std::vector<float>& final_features) {
    final_features = initial_features;
    
    float dt = integration_time / 100.0f;  // 100 integration steps
    
    for (int step = 0; step < 100; ++step) {
        if (integration_method == "euler") {
            euler_integration(graph, final_features, dt);
        } else if (integration_method == "rk4") {
            rk4_integration(graph, final_features, dt);
        }
    }
}

void GraphNeuralODE::dynamics(const GraphStructure& graph,
                             const std::vector<float>& features,
                             std::vector<float>& derivatives) const {
    size_t num_nodes = graph.num_nodes();
    derivatives.assign(num_nodes * feature_dim, 0.0f);
    
    // Compute graph-based dynamics
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& neighbors = graph.get_neighbors(i);
        
        // Self-dynamics
        for (size_t j = 0; j < feature_dim; ++j) {
            for (size_t k = 0; k < feature_dim; ++k) {
                derivatives[i * feature_dim + j] += features[i * feature_dim + k] * 
                                                     ode_weights[k * feature_dim + j];
            }
        }
        
        // Neighbor influence
        for (size_t neighbor : neighbors) {
            for (size_t j = 0; j < feature_dim; ++j) {
                for (size_t k = 0; k < feature_dim; ++k) {
                    derivatives[i * feature_dim + j] += features[neighbor * feature_dim + k] * 
                                                         ode_weights[k * feature_dim + j] * 0.1f;
                }
            }
        }
    }
}

void GraphNeuralODE::euler_integration(const GraphStructure& graph,
                                       std::vector<float>& features, float dt) const {
    std::vector<float> derivatives;
    dynamics(graph, features, derivatives);
    
    for (size_t i = 0; i < features.size(); ++i) {
        features[i] += dt * derivatives[i];
    }
}

void GraphNeuralODE::rk4_integration(const GraphStructure& graph,
                                    std::vector<float>& features, float dt) const {
    std::vector<float> k1, k2, k3, k4;
    std::vector<float> temp_features = features;
    
    // k1
    dynamics(graph, temp_features, k1);
    
    // k2
    for (size_t i = 0; i < features.size(); ++i) {
        temp_features[i] = features[i] + 0.5f * dt * k1[i];
    }
    dynamics(graph, temp_features, k2);
    
    // k3
    for (size_t i = 0; i < features.size(); ++i) {
        temp_features[i] = features[i] + 0.5f * dt * k2[i];
    }
    dynamics(graph, temp_features, k3);
    
    // k4
    for (size_t i = 0; i < features.size(); ++i) {
        temp_features[i] = features[i] + dt * k3[i];
    }
    dynamics(graph, temp_features, k4);
    
    // Update features
    for (size_t i = 0; i < features.size(); ++i) {
        features[i] += dt * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]) / 6.0f;
    }
}

// GNNFactory Implementation
std::unique_ptr<GraphConvolutionalLayer> GNNFactory::create_gcn(size_t in_dim, size_t out_dim) {
    return std::make_unique<GraphConvolutionalLayer>(in_dim, out_dim);
}

std::unique_ptr<GraphAttentionLayer> GNNFactory::create_gat(size_t in_dim, size_t out_dim, size_t heads) {
    return std::make_unique<GraphAttentionLayer>(in_dim, out_dim, heads);
}

std::unique_ptr<GraphSAGELayer> GNNFactory::create_graphsage(size_t in_dim, size_t out_dim, 
                                                             const std::string& agg_type) {
    return std::make_unique<GraphSAGELayer>(in_dim, out_dim, agg_type);
}

std::unique_ptr<TemporalGraphLayer> GNNFactory::create_temporal_gnn(size_t in_dim, size_t out_dim, 
                                                                    size_t time_steps) {
    return std::make_unique<TemporalGraphLayer>(in_dim, out_dim, time_steps);
}

std::unique_ptr<SpatiotemporalTransformer> GNNFactory::create_st_transformer(size_t spatial_dim, 
                                                                             size_t temporal_dim) {
    return std::make_unique<SpatiotemporalTransformer>(spatial_dim, temporal_dim);
}

std::unique_ptr<GraphNeuralODE> GNNFactory::create_gnn_ode(size_t feat_dim) {
    return std::make_unique<GraphNeuralODE>(feat_dim);
}

std::unique_ptr<HeterogeneousGraph> GNNFactory::create_heterogeneous_graph(size_t feat_dim) {
    return std::make_unique<HeterogeneousGraph>(feat_dim);
}

std::unique_ptr<ScalableGNN> GNNFactory::create_scalable_gnn(size_t n_nodes, size_t feat_dim) {
    return std::make_unique<ScalableGNN>(n_nodes, feat_dim);
}

std::unique_ptr<ScalableGNN> GNNFactory::create_scalable_gnn(size_t n_nodes, size_t feat_dim,
                                                             size_t batch_size, size_t num_neighbors) {
    return std::make_unique<ScalableGNN>(n_nodes, feat_dim, batch_size, num_neighbors);
}

std::unique_ptr<GeometricDeepLearning> GNNFactory::create_geometric_gnn(size_t manifold_dim, 
                                                                       size_t feat_dim) {
    return std::make_unique<GeometricDeepLearning>(manifold_dim, feat_dim);
}

// HeterogeneousGraph Implementation
void HeterogeneousGraph::add_node_type(const std::string& type, const std::vector<float>& embedding) {
    type_embeddings[type] = embedding;
}

void HeterogeneousGraph::add_relation_type(const std::string& source_type, const std::string& target_type,
                                          const std::vector<float>& embedding) {
    relation_embeddings[std::make_pair(source_type, target_type)] = embedding;
}

void HeterogeneousGraph::heterogeneous_message_passing(const std::vector<float>& input_features,
                                                      std::vector<float>& output_features) {
    size_t num_nodes = this->num_nodes();
    output_features.assign(num_nodes * get_feature_dim(), 0.0f);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = get_node(i);
        const auto& neighbors = get_neighbors(i);
        
        // Get type embedding for current node
        auto type_it = type_embeddings.find(node.type);
        if (type_it != type_embeddings.end()) {
            const auto& type_emb = type_it->second;
            
            // Aggregate messages from neighbors with type-specific weights
            for (size_t neighbor : neighbors) {
                const auto& neighbor_node = get_node(neighbor);
                auto relation_it = relation_embeddings.find(std::make_pair(neighbor_node.type, node.type));
                
                if (relation_it != relation_embeddings.end()) {
                    const auto& relation_emb = relation_it->second;
                    
                    // Apply relation-specific transformation
                    for (size_t j = 0; j < get_feature_dim(); ++j) {
                        for (size_t k = 0; k < relation_emb.size(); ++k) {
                            output_features[i * get_feature_dim() + j] += 
                                input_features[neighbor * get_feature_dim() + k] * relation_emb[k];
                        }
                    }
                }
            }
            
            // Apply type-specific transformation
            for (size_t j = 0; j < get_feature_dim(); ++j) {
                for (size_t k = 0; k < type_emb.size(); ++k) {
                    output_features[i * get_feature_dim() + j] += 
                        input_features[i * get_feature_dim() + k] * type_emb[k];
                }
            }
        }
    }
}

void HeterogeneousGraph::metapath_aggregation(const std::vector<std::string>& metapath,
                                              const std::vector<float>& input_features,
                                              std::vector<float>& output_features) {
    if (metapath.size() < 2) return;
    
    size_t num_nodes = this->num_nodes();
    std::vector<float> temp_features = input_features;
    
    // Follow metapath through the graph
    for (size_t step = 0; step < metapath.size() - 1; ++step) {
        const std::string& current_type = metapath[step];
        const std::string& next_type = metapath[step + 1];
        
        // Aggregate features from nodes of current_type to nodes of next_type
        std::vector<float> aggregated(num_nodes * get_feature_dim(), 0.0f);
        
        for (size_t i = 0; i < num_nodes; ++i) {
            const auto& node = get_node(i);
            if (node.type == next_type) {
                const auto& neighbors = get_neighbors(i);
                
                for (size_t neighbor : neighbors) {
                    const auto& neighbor_node = get_node(neighbor);
                    if (neighbor_node.type == current_type) {
                        for (size_t j = 0; j < get_feature_dim(); ++j) {
                            aggregated[i * get_feature_dim() + j] += 
                                temp_features[neighbor * get_feature_dim() + j];
                        }
                    }
                }
                
                // Normalize by number of matching neighbors
                size_t matching_neighbors = 0;
                for (size_t neighbor : neighbors) {
                    if (get_node(neighbor).type == current_type) {
                        matching_neighbors++;
                    }
                }
                
                if (matching_neighbors > 0) {
                    for (size_t j = 0; j < get_feature_dim(); ++j) {
                        aggregated[i * get_feature_dim() + j] /= matching_neighbors;
                    }
                }
            }
        }
        
        temp_features = aggregated;
    }
    
    output_features = temp_features;
}

// ScalableGNN Implementation
ScalableGNN::ScalableGNN(size_t n_nodes, size_t feat_dim, size_t batch_sz, size_t n_neighbors)
    : num_nodes(n_nodes), feature_dim(feat_dim), batch_size(batch_sz), num_neighbors(n_neighbors) {}

void ScalableGNN::add_layer(std::unique_ptr<MessagePassingLayer> layer) {
    layers.push_back(std::move(layer));
}

void ScalableGNN::neighbor_sampling(const GraphStructure& graph, size_t target_node,
                                    size_t num_samples, std::vector<size_t>& sampled_nodes) const {
    const auto& neighbors = graph.get_neighbors(target_node);
    sampled_nodes.clear();
    
    if (neighbors.size() <= num_samples) {
        sampled_nodes = neighbors;
    } else {
        // Random sampling
        std::vector<size_t> temp_neighbors = neighbors;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(temp_neighbors.begin(), temp_neighbors.end(), gen);
        
        for (size_t i = 0; i < num_samples; ++i) {
            sampled_nodes.push_back(temp_neighbors[i]);
        }
    }
    
    // Always include the target node itself
    sampled_nodes.push_back(target_node);
}

void ScalableGNN::layer_sampling(const GraphStructure& graph, const std::vector<size_t>& target_nodes,
                                 std::vector<std::vector<size_t>>& sampled_layers) const {
    sampled_layers.clear();
    sampled_layers.resize(layers.size());
    
    // Sample for each layer
    std::vector<size_t> current_batch = target_nodes;
    
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        sampled_layers[layer_idx] = current_batch;
        
        std::vector<size_t> next_batch;
        for (size_t node : current_batch) {
            std::vector<size_t> sampled_neighbors;
            neighbor_sampling(graph, node, num_neighbors, sampled_neighbors);
            next_batch.insert(next_batch.end(), sampled_neighbors.begin(), sampled_neighbors.end());
        }
        
        // Remove duplicates and update for next layer
        std::unordered_set<size_t> unique_nodes(next_batch.begin(), next_batch.end());
        current_batch.assign(unique_nodes.begin(), unique_nodes.end());
    }
}

void ScalableGNN::cluster_sampling(const GraphStructure& graph, size_t num_clusters,
                                   std::vector<std::vector<size_t>>& clusters) const {
    clusters.clear();
    clusters.assign(num_clusters, std::vector<size_t>());
    
    // Simple random clustering for demonstration
    std::vector<size_t> all_nodes(num_nodes);
    std::iota(all_nodes.begin(), all_nodes.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(all_nodes.begin(), all_nodes.end(), gen);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        clusters[i % num_clusters].push_back(all_nodes[i]);
    }
}

void ScalableGNN::train_step(const GraphStructure& graph,
                             const std::vector<float>& features,
                             const std::vector<float>& targets,
                             const std::vector<size_t>& sampled_nodes) {
    std::vector<float> current_features;
    for (size_t node : sampled_nodes) {
        current_features.insert(current_features.end(),
                               features.begin() + node * feature_dim,
                               features.begin() + (node + 1) * feature_dim);
    }
    
    // Forward pass through layers
    for (const auto& layer : layers) {
        std::vector<float> layer_output;
        
        // Create subgraph with sampled nodes
        // This is simplified - in practice you'd need to handle edge sampling too
        layer->forward(graph, current_features, layer_output);
        current_features = layer_output;
    }
    
    // Compute loss and update weights (simplified)
    // In practice, you'd implement proper backpropagation here
}

void ScalableGNN::forward_sampled(const GraphStructure& graph,
                                  const std::vector<float>& features,
                                  const std::vector<size_t>& sampled_nodes,
                                  std::vector<float>& output) {
    std::unordered_map<size_t, size_t> node_map;
    node_map.reserve(sampled_nodes.size());
    for (size_t i = 0; i < sampled_nodes.size(); ++i) {
        node_map[sampled_nodes[i]] = i;
    }

    GraphStructure subgraph(feature_dim);
    std::vector<float> current_features;
    current_features.reserve(sampled_nodes.size() * feature_dim);

    for (size_t node : sampled_nodes) {
        const auto& node_data = graph.get_node(node);
        subgraph.add_node(node_data.features, node_data.type);
        current_features.insert(current_features.end(),
                               features.begin() + node * feature_dim,
                               features.begin() + (node + 1) * feature_dim);
    }

    for (const auto& edge : graph.get_edges()) {
        auto src_it = node_map.find(edge.source);
        auto tgt_it = node_map.find(edge.target);
        if (src_it != node_map.end() && tgt_it != node_map.end()) {
            subgraph.add_edge(src_it->second, tgt_it->second, edge.weight, edge.type);
        }
    }
    
    // Forward pass through layers
    for (const auto& layer : layers) {
        std::vector<float> layer_output;
        layer->forward(subgraph, current_features, layer_output);
        current_features = layer_output;
    }
    
    output = current_features;
}

// GeometricDeepLearning Implementation
GeometricDeepLearning::GeometricDeepLearning(size_t manifold_d, size_t feat_dim)
    : manifold_dim(manifold_d), feature_dim(feat_dim) {
    
    metric_tensor.assign(manifold_dim * manifold_dim, 0.0f);
    connection_coefficients.assign(manifold_dim * manifold_dim * manifold_dim, 0.0f);
    
    // Initialize metric tensor (identity matrix for simplicity)
    for (size_t i = 0; i < manifold_dim; ++i) {
        metric_tensor[i * manifold_dim + i] = 1.0f;
    }
}

void GeometricDeepLearning::compute_geodesic_distances(const GraphStructure& graph,
                                                      std::vector<std::vector<float>>& distances) const {
    size_t num_nodes = graph.num_nodes();
    distances.assign(num_nodes, std::vector<float>(num_nodes, 0.0f));
    
    // Simplified geodesic distance computation using graph shortest paths
    for (size_t i = 0; i < num_nodes; ++i) {
        for (size_t j = 0; j < num_nodes; ++j) {
            if (i == j) {
                distances[i][j] = 0.0f;
            } else {
                // Use feature distance as proxy for geodesic distance
                const auto& node_i = graph.get_node(i);
                const auto& node_j = graph.get_node(j);
                
                float dist = 0.0f;
                for (size_t k = 0; k < node_i.features.size(); ++k) {
                    float diff = node_i.features[k] - node_j.features[k];
                    dist += diff * diff;
                }
                distances[i][j] = std::sqrt(dist);
            }
        }
    }
}

void GeometricDeepLearning::manifold_attention(const GraphStructure& graph,
                                               const std::vector<float>& features,
                                               std::vector<float>& output) {
    size_t num_nodes = graph.num_nodes();
    output.assign(num_nodes * feature_dim, 0.0f);
    
    // Compute geodesic distances
    std::vector<std::vector<float>> distances;
    compute_geodesic_distances(graph, distances);
    
    // Apply attention based on geodesic distances
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& neighbors = graph.get_neighbors(i);
        
        // Compute attention weights
        std::vector<float> attention_weights;
        float sum_weights = 0.0f;
        
        for (size_t neighbor : neighbors) {
            float dist = distances[i][neighbor];
            float weight = std::exp(-dist * dist);  // Gaussian kernel
            attention_weights.push_back(weight);
            sum_weights += weight;
        }
        
        // Normalize weights
        for (auto& weight : attention_weights) {
            weight /= sum_weights;
        }
        
        // Aggregate features with attention weights
        for (size_t idx = 0; idx < neighbors.size(); ++idx) {
            size_t neighbor = neighbors[idx];
            float weight = attention_weights[idx];
            
            for (size_t j = 0; j < feature_dim; ++j) {
                output[i * feature_dim + j] += weight * features[neighbor * feature_dim + j];
            }
        }
    }
}

void GeometricDeepLearning::riemannian_gradient(const std::vector<float>& features,
                                               const std::vector<float>& gradients,
                                               std::vector<float>& riemannian_grad) const {
    size_t num_elements = features.size();
    riemannian_grad.assign(num_elements, 0.0f);
    
    // Simplified Riemannian gradient computation
    // In practice, this would involve the metric tensor and Christoffel symbols
    for (size_t i = 0; i < num_elements; ++i) {
        riemannian_grad[i] = gradients[i];
        
        // Apply metric tensor correction (simplified)
        for (size_t j = 0; j < manifold_dim && j < num_elements; ++j) {
            riemannian_grad[i] += metric_tensor[j * manifold_dim + (j % manifold_dim)] * gradients[j];
        }
    }
}

void GeometricDeepLearning::parallel_transport(const std::vector<float>& source_point,
                                              const std::vector<float>& target_point,
                                              const std::vector<float>& vector,
                                              std::vector<float>& transported) const {
    // Simplified parallel transport
    // In practice, this would involve solving differential equations along geodesics
    
    transported = vector;
    
    // Apply correction based on manifold curvature (simplified)
    for (size_t i = 0; i < vector.size(); ++i) {
        if (i < source_point.size() && i < target_point.size()) {
            float diff = target_point[i] - source_point[i];
            transported[i] += diff * 0.1f;  // Simple linear correction
        }
    }
}

void GeometricDeepLearning::exponential_map(const std::vector<float>& base_point,
                                           const std::vector<float>& tangent_vector,
                                           std::vector<float>& result) const {
    // Simplified exponential map
    result = base_point;
    
    for (size_t i = 0; i < tangent_vector.size() && i < result.size(); ++i) {
        result[i] += tangent_vector[i];
    }
}

void GeometricDeepLearning::logarithmic_map(const std::vector<float>& base_point,
                                            const std::vector<float>& target_point,
                                            std::vector<float>& result) const {
    // Simplified logarithmic map
    result.resize(target_point.size());
    
    for (size_t i = 0; i < target_point.size() && i < base_point.size(); ++i) {
        result[i] = target_point[i] - base_point[i];
    }
}

} // namespace GNN
} // namespace ML
