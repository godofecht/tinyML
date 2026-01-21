//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Spatiotemporal & Graph Neural Networks Framework
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 2025
*****************************************************************************/

#ifndef GRAPH_NEURAL_NETWORK_H
#define GRAPH_NEURAL_NETWORK_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <algorithm>
#include <random>
#include <cmath>
#include <xsimd/xsimd.hpp>
#include "XSIMDOperations.h"

namespace ML {
namespace GNN {

// Forward declarations
class GraphStructure;
class MessagePassingLayer;
class GraphConvolutionalLayer;
class GraphAttentionLayer;
class TemporalGraphLayer;
class SpatiotemporalTransformer;
class GraphNeuralODE;
class HeterogeneousGraph;
class ScalableGNN;
class GeometricDeepLearning;

// Graph data structures
struct Node {
    size_t id;
    std::vector<float> features;
    std::vector<size_t> neighbors;
    std::string type;  // For heterogeneous graphs
    std::unordered_map<std::string, float> attributes;
    
    Node(size_t node_id, const std::vector<float>& node_features, const std::string& node_type = "")
        : id(node_id), features(node_features), type(node_type) {}
};

struct Edge {
    size_t source;
    size_t target;
    float weight;
    std::string type;  // For heterogeneous graphs
    std::unordered_map<std::string, float> attributes;
    
    Edge(size_t src, size_t tgt, float w = 1.0f, const std::string& edge_type = "")
        : source(src), target(tgt), weight(w), type(edge_type) {}
};

class GraphStructure {
private:
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::unordered_map<size_t, std::vector<size_t>> adjacency_list;
    std::unordered_map<std::string, std::vector<size_t>> node_type_mapping;
    std::unordered_map<std::string, std::vector<size_t>> edge_type_mapping;
    bool is_directed;
    size_t feature_dim;

public:
    GraphStructure(size_t feat_dim, bool directed = false)
        : feature_dim(feat_dim), is_directed(directed) {}
    
    // Graph construction
    size_t add_node(const std::vector<float>& features, const std::string& type = "");
    void add_edge(size_t source, size_t target, float weight = 1.0f, const std::string& type = "");
    void remove_node(size_t node_id);
    void remove_edge(size_t source, size_t target);
    
    // Graph access
    const Node& get_node(size_t node_id) const;
    const Edge& get_edge(size_t edge_id) const;
    const std::vector<size_t>& get_neighbors(size_t node_id) const;
    const std::vector<Node>& get_nodes() const { return nodes; }
    const std::vector<Edge>& get_edges() const { return edges; }
    
    // Graph properties
    size_t num_nodes() const { return nodes.size(); }
    size_t num_edges() const { return edges.size(); }
    size_t get_feature_dim() const { return feature_dim; }
    bool empty() const { return nodes.empty(); }
    
    // Heterogeneous graph support
    const std::vector<size_t>& get_nodes_by_type(const std::string& type) const;
    const std::vector<size_t>& get_edges_by_type(const std::string& type) const;
    std::vector<std::string> get_node_types() const;
    std::vector<std::string> get_edge_types() const;
    
    // Graph utilities
    void normalize_features();
    void compute_laplacian(std::vector<float>& laplacian) const;
    std::vector<float> get_degree_matrix() const;
    std::vector<float> get_adjacency_matrix() const;
    void clear();
};

// Message Passing Neural Network Framework
class MessagePassingLayer {
protected:
    size_t input_dim;
    size_t output_dim;
    std::vector<float> message_weights;
    std::vector<float> update_weights;
    std::vector<float> message_bias;
    std::vector<float> update_bias;
    float dropout_rate;
    bool use_batch_norm;
    
public:
    MessagePassingLayer(size_t in_dim, size_t out_dim, float dropout = 0.0f, bool batch_norm = false);
    virtual ~MessagePassingLayer() = default;
    
    // Core message passing interface
    virtual void message_function(const GraphStructure& graph, 
                                 const std::vector<float>& node_features,
                                 std::vector<std::vector<float>>& messages) const = 0;
    virtual void aggregate_function(const GraphStructure& graph,
                                   const std::vector<std::vector<float>>& messages,
                                   std::vector<float>& aggregated) const = 0;
    virtual void update_function(const std::vector<float>& aggregated,
                               std::vector<float>& output) const = 0;
    
    // Forward pass
    void forward(const GraphStructure& graph, 
                const std::vector<float>& input_features,
                std::vector<float>& output_features);
    
    // Utility methods
    size_t get_input_dim() const { return input_dim; }
    size_t get_output_dim() const { return output_dim; }
    void set_dropout(float rate) { dropout_rate = rate; }
    
protected:
    void apply_activation(std::vector<float>& x) const;
    void apply_batch_norm(std::vector<float>& x) const;
    void apply_dropout(std::vector<float>& x, std::mt19937& rng) const;
};

// Graph Convolutional Network (GCN)
class GraphConvolutionalLayer : public MessagePassingLayer {
private:
    bool normalize_adjacency;
    bool use_self_loops;
    
public:
    GraphConvolutionalLayer(size_t in_dim, size_t out_dim, 
                           bool normalize_adj = true, bool self_loops = true,
                           float dropout = 0.0f, bool batch_norm = false);
    
    void message_function(const GraphStructure& graph, 
                         const std::vector<float>& node_features,
                         std::vector<std::vector<float>>& messages) const override;
    
    void aggregate_function(const GraphStructure& graph,
                           const std::vector<std::vector<float>>& messages,
                           std::vector<float>& aggregated) const override;
    
    void update_function(const std::vector<float>& aggregated,
                        std::vector<float>& output) const override;
};

// Graph Attention Network (GAT)
class GraphAttentionLayer : public MessagePassingLayer {
private:
    size_t num_heads;
    std::vector<float> attention_weights;
    float negative_slope;
    bool use_residual;
    
public:
    GraphAttentionLayer(size_t in_dim, size_t out_dim, size_t heads = 8,
                       float neg_slope = 0.2f, bool residual = false,
                       float dropout = 0.0f, bool batch_norm = false);
    
    void message_function(const GraphStructure& graph, 
                         const std::vector<float>& node_features,
                         std::vector<std::vector<float>>& messages) const override;
    
    void aggregate_function(const GraphStructure& graph,
                           const std::vector<std::vector<float>>& messages,
                           std::vector<float>& aggregated) const override;
    
    void update_function(const std::vector<float>& aggregated,
                        std::vector<float>& output) const override;
    
private:
    std::vector<float> compute_attention_scores(const std::vector<float>& source_feat,
                                              const std::vector<float>& target_feat) const;
    void multi_head_attention(const std::vector<std::vector<float>>& messages,
                             std::vector<float>& aggregated) const;
};

// GraphSAGE (Sample and AggreGatE)
class GraphSAGELayer : public MessagePassingLayer {
private:
    std::string aggregator_type;  // "mean", "max", "lstm"
    size_t num_samples;
    bool normalize_features;
    
public:
    GraphSAGELayer(size_t in_dim, size_t out_dim, 
                   const std::string& agg_type = "mean",
                   size_t samples = 10, bool normalize = false,
                   float dropout = 0.0f, bool batch_norm = false);
    
    void message_function(const GraphStructure& graph, 
                         const std::vector<float>& node_features,
                         std::vector<std::vector<float>>& messages) const override;
    
    void aggregate_function(const GraphStructure& graph,
                           const std::vector<std::vector<float>>& messages,
                           std::vector<float>& aggregated) const override;
    
    void update_function(const std::vector<float>& aggregated,
                        std::vector<float>& output) const override;
    
private:
    void mean_aggregation(const std::vector<std::vector<float>>& messages,
                         std::vector<float>& aggregated) const;
    void max_aggregation(const std::vector<std::vector<float>>& messages,
                        std::vector<float>& aggregated) const;
    void lstm_aggregation(const std::vector<std::vector<float>>& messages,
                         std::vector<float>& aggregated) const;
};

// Temporal Graph Network
class TemporalGraphLayer : public MessagePassingLayer {
private:
    size_t time_steps;
    std::vector<float> temporal_weights;
    float time_decay_factor;
    bool use_gru_update;
    
public:
    TemporalGraphLayer(size_t in_dim, size_t out_dim, size_t t_steps = 10,
                      float decay = 0.9f, bool gru = true,
                      float dropout = 0.0f, bool batch_norm = false);
    
    void message_function(const GraphStructure& graph, 
                         const std::vector<float>& node_features,
                         std::vector<std::vector<float>>& messages) const override;
    
    void aggregate_function(const GraphStructure& graph,
                           const std::vector<std::vector<float>>& messages,
                           std::vector<float>& aggregated) const override;
    
    void update_function(const std::vector<float>& aggregated,
                        std::vector<float>& output) const override;
    
    // Temporal-specific methods
    void forward_temporal(const std::vector<GraphStructure>& graph_sequence,
                         const std::vector<std::vector<float>>& feature_sequence,
                         std::vector<float>& output_features);
    
private:
    void apply_temporal_decay(std::vector<float>& features, size_t time_step) const;
    void gru_update(const std::vector<float>& input, const std::vector<float>& hidden,
                   std::vector<float>& output) const;
};

// Spatiotemporal Transformer
class SpatiotemporalTransformer {
private:
    size_t spatial_dim;
    size_t temporal_dim;
    size_t num_heads;
    size_t num_layers;
    std::vector<float> spatial_attention_weights;
    std::vector<float> temporal_attention_weights;
    std::vector<float> fusion_weights;
    float dropout_rate;
    
public:
    SpatiotemporalTransformer(size_t spatial_d, size_t temporal_d, 
                              size_t heads = 8, size_t layers = 3,
                              float dropout = 0.1f);
    
    void forward(const GraphStructure& graph,
                const std::vector<std::vector<float>>& temporal_features,
                std::vector<float>& output);
    
    // Spatial attention
    void spatial_attention(const GraphStructure& graph,
                          const std::vector<float>& features,
                          std::vector<float>& spatial_output);
    
    // Temporal attention
    void temporal_attention(const std::vector<std::vector<float>>& temporal_seq,
                           std::vector<float>& temporal_output);
    
    // Space-time fusion
    void spatiotemporal_fusion(const std::vector<float>& spatial_feat,
                               const std::vector<float>& temporal_feat,
                               std::vector<float>& fused_output);
    
private:
    void compute_spatial_attention_weights(const GraphStructure& graph,
                                          const std::vector<float>& features);
    void compute_temporal_attention_weights(const std::vector<std::vector<float>>& temporal_seq);
};

// Graph Neural ODE
class GraphNeuralODE {
private:
    size_t feature_dim;
    std::vector<float> ode_weights;
    float integration_time;
    std::string integration_method;  // "euler", "rk4", "dopri5"
    
public:
    GraphNeuralODE(size_t feat_dim, float int_time = 1.0f,
                   const std::string& method = "rk4");
    
    void forward(const GraphStructure& graph,
                const std::vector<float>& initial_features,
                std::vector<float>& final_features);
    
    // ODE dynamics
    void dynamics(const GraphStructure& graph,
                const std::vector<float>& features,
                std::vector<float>& derivatives) const;
    
    // Integration methods
    void euler_integration(const GraphStructure& graph,
                          std::vector<float>& features, float dt) const;
    void rk4_integration(const GraphStructure& graph,
                        std::vector<float>& features, float dt) const;
};

// Heterogeneous Graph Support
// Custom hash function for std::pair<std::string, std::string>
struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1); // Simple hash combination
    }
};

class HeterogeneousGraph : public GraphStructure {
private:
    std::unordered_map<std::string, std::vector<float>> type_embeddings;
    std::unordered_map<std::pair<std::string, std::string>, std::vector<float>, 
                       PairHash> relation_embeddings;
    
public:
    HeterogeneousGraph(size_t feat_dim, bool directed = false)
        : GraphStructure(feat_dim, directed) {}
    
    // Heterogeneous operations
    void add_node_type(const std::string& type, const std::vector<float>& embedding);
    void add_relation_type(const std::string& source_type, const std::string& target_type,
                          const std::vector<float>& embedding);
    
    void heterogeneous_message_passing(const std::vector<float>& input_features,
                                      std::vector<float>& output_features);
    
    void metapath_aggregation(const std::vector<std::string>& metapath,
                             const std::vector<float>& input_features,
                             std::vector<float>& output_features);
};

// Scalable GNN with Sampling
class ScalableGNN {
private:
    size_t num_nodes;
    size_t feature_dim;
    size_t batch_size;
    size_t num_neighbors;
    std::vector<std::unique_ptr<MessagePassingLayer>> layers;
    
public:
    ScalableGNN(size_t n_nodes, size_t feat_dim, size_t batch_sz = 1024,
                size_t n_neighbors = 10);
    
    void add_layer(std::unique_ptr<MessagePassingLayer> layer);
    
    // Sampling methods
    void neighbor_sampling(const GraphStructure& graph, size_t target_node,
                          size_t num_samples, std::vector<size_t>& sampled_nodes) const;
    
    void layer_sampling(const GraphStructure& graph, const std::vector<size_t>& target_nodes,
                       std::vector<std::vector<size_t>>& sampled_layers) const;
    
    void cluster_sampling(const GraphStructure& graph, size_t num_clusters,
                         std::vector<std::vector<size_t>>& clusters) const;
    
    // Training with sampling
    void train_step(const GraphStructure& graph,
                   const std::vector<float>& features,
                   const std::vector<float>& targets,
                   const std::vector<size_t>& sampled_nodes);
    
    void forward_sampled(const GraphStructure& graph,
                        const std::vector<float>& features,
                        const std::vector<size_t>& sampled_nodes,
                        std::vector<float>& output);
};

// Geometric Deep Learning
class GeometricDeepLearning {
private:
    size_t manifold_dim;
    size_t feature_dim;
    std::vector<float> metric_tensor;
    std::vector<float> connection_coefficients;
    
public:
    GeometricDeepLearning(size_t manifold_d, size_t feat_dim);
    
    // Manifold learning operations
    void compute_geodesic_distances(const GraphStructure& graph,
                                   std::vector<std::vector<float>>& distances) const;
    
    void manifold_attention(const GraphStructure& graph,
                           const std::vector<float>& features,
                           std::vector<float>& output);
    
    void riemannian_gradient(const std::vector<float>& features,
                            const std::vector<float>& gradients,
                            std::vector<float>& riemannian_grad) const;
    
    void parallel_transport(const std::vector<float>& source_point,
                          const std::vector<float>& target_point,
                          const std::vector<float>& vector,
                          std::vector<float>& transported) const;
    
    // Geometric operations
    void exponential_map(const std::vector<float>& base_point,
                        const std::vector<float>& tangent_vector,
                        std::vector<float>& result) const;
    
    void logarithmic_map(const std::vector<float>& base_point,
                        const std::vector<float>& target_point,
                        std::vector<float>& result) const;
};

// Factory for creating GNN models
class GNNFactory {
public:
    static std::unique_ptr<GraphConvolutionalLayer> create_gcn(size_t in_dim, size_t out_dim);
    static std::unique_ptr<GraphAttentionLayer> create_gat(size_t in_dim, size_t out_dim, size_t heads = 8);
    static std::unique_ptr<GraphSAGELayer> create_graphsage(size_t in_dim, size_t out_dim, 
                                                           const std::string& agg_type = "mean");
    static std::unique_ptr<TemporalGraphLayer> create_temporal_gnn(size_t in_dim, size_t out_dim, 
                                                                   size_t time_steps = 10);
    static std::unique_ptr<SpatiotemporalTransformer> create_st_transformer(size_t spatial_dim, 
                                                                            size_t temporal_dim);
    static std::unique_ptr<GraphNeuralODE> create_gnn_ode(size_t feat_dim);
    static std::unique_ptr<HeterogeneousGraph> create_heterogeneous_graph(size_t feat_dim);
    static std::unique_ptr<ScalableGNN> create_scalable_gnn(size_t n_nodes, size_t feat_dim);
    static std::unique_ptr<ScalableGNN> create_scalable_gnn(size_t n_nodes, size_t feat_dim,
                                                            size_t batch_size, size_t num_neighbors);
    static std::unique_ptr<GeometricDeepLearning> create_geometric_gnn(size_t manifold_dim, 
                                                                       size_t feat_dim);
};

} // namespace GNN
} // namespace ML

#endif // GRAPH_NEURAL_NETWORK_H
