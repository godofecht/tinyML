#ifndef REINFORCEMENTLEARNING_H
#define REINFORCEMENTLEARNING_H

#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <deque>
#include "XSIMDOperations.h"
#include "DynamicNeuralNetwork.h"

namespace TinyML {

using ML::Dynamic::DynamicNeuralNetwork;

// Forward declarations
class Environment;
class ReplayBuffer;
class NeuralNetwork;

// Basic RL data structures
struct State {
    std::vector<float> features;
    bool terminal = false;
    
    State() = default;
    State(const std::vector<float>& f) : features(f) {}
    State(size_t size) : features(size, 0.0f) {}
};

struct Action {
    int action_id;
    std::vector<float> continuous;
    
    Action() : action_id(0) {}
    Action(int id) : action_id(id) {}
    Action(const std::vector<float>& cont) : continuous(cont), action_id(-1) {}
};

struct Transition {
    State state;
    Action action;
    float reward;
    State next_state;
    bool done;
    
    Transition(const State& s, const Action& a, float r, const State& ns, bool d)
        : state(s), action(a), reward(r), next_state(ns), done(d) {}
};

// Environment interface
class Environment {
public:
    virtual ~Environment() = default;
    virtual State reset() = 0;
    virtual std::pair<State, float> step(const Action& action) = 0;
    virtual int get_action_space_size() const = 0;
    virtual int get_state_space_size() const = 0;
    virtual bool is_discrete() const = 0;
};

// Replay Buffer for experience replay
class ReplayBuffer {
private:
    std::deque<Transition> buffer;
    size_t capacity;
    std::mt19937 rng;
    
public:
    ReplayBuffer(size_t cap = 100000) : capacity(cap), rng(std::random_device{}()) {}
    
    void push(const Transition& transition) {
        if (buffer.size() >= capacity) {
            buffer.pop_front();
        }
        buffer.push_back(transition);
    }
    
    std::vector<Transition> sample(size_t batch_size) {
        std::vector<Transition> batch;
        std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
        
        for (size_t i = 0; i < batch_size && i < buffer.size(); ++i) {
            size_t idx = dist(rng);
            batch.push_back(buffer[idx]);
        }
        return batch;
    }
    
    size_t size() const { return buffer.size(); }
};

// Base RL Agent class
class RLAgent {
protected:
    std::mt19937 rng;
    float learning_rate;
    float gamma;
    int state_dim;
    int action_dim;
    
public:
    RLAgent(int state_dim, int action_dim, float lr = 0.001, float gamma = 0.99)
        : rng(std::random_device{}()), learning_rate(lr), gamma(gamma), 
          state_dim(state_dim), action_dim(action_dim) {}
    
    virtual ~RLAgent() = default;
    virtual Action select_action(const State& state, bool explore = true) = 0;
    virtual void train_step(const std::vector<Transition>& batch) = 0;
    virtual void update_target_network() {}
    virtual float get_epsilon() const { return 0.0f; }
    int get_state_dim() const { return state_dim; }
    int get_action_dim() const { return action_dim; }
    float get_learning_rate() const { return learning_rate; }
    float get_gamma() const { return gamma; }
};

// Deep Q-Network (DQN)
class DQN : public RLAgent {
protected:
    std::unique_ptr<DynamicNeuralNetwork> q_network;
    std::unique_ptr<DynamicNeuralNetwork> target_network;
    float epsilon;
    float epsilon_decay;
    float epsilon_min;
    int update_frequency;
    int steps_since_update;
    
public:
    DQN(int state_dim, int action_dim, const std::vector<int>& hidden_layers = {128, 64},
        float lr = 0.001, float gamma = 0.99, float epsilon_start = 1.0f,
        float epsilon_decay = 0.995, float epsilon_min = 0.01f);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
    void update_target_network() override;
    float get_epsilon() const override { return epsilon; }
    
    virtual std::vector<float> get_q_values(const State& state);
};

// Double DQN
class DoubleDQN : public DQN {
public:
    DoubleDQN(int state_dim, int action_dim, const std::vector<int>& hidden_layers = {128, 64},
              float lr = 0.001, float gamma = 0.99, float epsilon_start = 1.0f,
              float epsilon_decay = 0.995, float epsilon_min = 0.01f);
    
    void train_step(const std::vector<Transition>& batch) override;
};

// Dueling DQN
class DuelingDQN : public DQN {
private:
    std::unique_ptr<DynamicNeuralNetwork> value_stream;
    std::unique_ptr<DynamicNeuralNetwork> advantage_stream;
    
public:
    DuelingDQN(int state_dim, int action_dim, const std::vector<int>& hidden_layers = {128, 64},
                float lr = 0.001, float gamma = 0.99, float epsilon_start = 1.0f,
                float epsilon_decay = 0.995, float epsilon_min = 0.01f);
    
    std::vector<float> get_q_values(const State& state) override;
};

// Policy Gradient Methods
class REINFORCE : public RLAgent {
private:
    std::unique_ptr<DynamicNeuralNetwork> policy_network;
    std::vector<float> log_probs;
    std::vector<float> rewards;
    
public:
    REINFORCE(int state_dim, int action_dim, const std::vector<int>& hidden_layers = {128, 64},
              float lr = 0.001, float gamma = 0.99);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
    void store_reward(float reward) { rewards.push_back(reward); }
    void update_policy();
};

// Advantage Actor-Critic (A2C)
class A2C : public RLAgent {
private:
    std::unique_ptr<DynamicNeuralNetwork> actor_network;
    std::unique_ptr<DynamicNeuralNetwork> critic_network;
    float entropy_coefficient;
    
public:
    A2C(int state_dim, int action_dim, const std::vector<int>& hidden_layers = {128, 64},
        float lr = 0.001, float gamma = 0.99, float entropy_coeff = 0.01f);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
    
    std::pair<float, float> evaluate_state(const State& state);
};

// Asynchronous Advantage Actor-Critic (A3C)
class A3C : public A2C {
private:
    int num_workers;
    std::vector<std::unique_ptr<A2C>> workers;
    
public:
    A3C(int state_dim, int action_dim, int num_workers = 4,
        const std::vector<int>& hidden_layers = {128, 64},
        float lr = 0.001, float gamma = 0.99, float entropy_coeff = 0.01f);
    
    void train_step(const std::vector<Transition>& batch) override;
    void train_parallel(Environment& env, int max_episodes = 1000);
};

// Proximal Policy Optimization (PPO)
class PPO : public RLAgent {
private:
    std::unique_ptr<DynamicNeuralNetwork> actor_network;
    std::unique_ptr<DynamicNeuralNetwork> critic_network;
    std::unique_ptr<DynamicNeuralNetwork> old_actor_network;
    float clip_epsilon;
    float entropy_coefficient;
    int ppo_epochs;
    int mini_batch_size;
    
public:
    PPO(int state_dim, int action_dim, const std::vector<int>& hidden_layers = {128, 64},
        float lr = 0.001, float gamma = 0.99, float clip_epsilon = 0.2f,
        float entropy_coeff = 0.01f, int ppo_epochs = 4, int mini_batch_size = 64);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
    void update_old_policy();
    
    float compute_ppo_loss(const std::vector<Transition>& batch);
};

// Trust Region Policy Optimization (TRPO)
class TRPO : public RLAgent {
private:
    std::unique_ptr<DynamicNeuralNetwork> actor_network;
    std::unique_ptr<DynamicNeuralNetwork> critic_network;
    float delta;  // KL divergence constraint
    int cg_iterations;  // Conjugate gradient iterations
    
public:
    TRPO(int state_dim, int action_dim, const std::vector<int>& hidden_layers = {128, 64},
         float lr = 0.001, float gamma = 0.99, float delta = 0.01f, int cg_iters = 10);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
    
    std::vector<float> conjugate_gradient(const std::vector<float>& grad, 
                                         const std::vector<Transition>& batch);
    float compute_kl_divergence(const std::vector<Transition>& batch);
};

// Soft Actor-Critic (SAC)
class SAC : public RLAgent {
private:
    std::unique_ptr<DynamicNeuralNetwork> actor_network;
    std::unique_ptr<DynamicNeuralNetwork> q1_network;
    std::unique_ptr<DynamicNeuralNetwork> q2_network;
    std::unique_ptr<DynamicNeuralNetwork> target_q_network;
    float alpha;  // Temperature parameter
    float target_entropy;
    bool auto_entropy_tuning;
    
public:
    SAC(int state_dim, int action_dim, const std::vector<int>& hidden_layers = {128, 64},
        float lr = 0.001, float gamma = 0.99, float alpha = 0.2f, 
        bool auto_entropy_tuning = true);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
    void update_target_network() override;
    
    float compute_q_value(const State& state, const Action& action, int network_id);
    std::pair<Action, float> sample_action_and_log_prob(const State& state);
};

// Model-Based RL - World Model
class WorldModel {
private:
    std::unique_ptr<DynamicNeuralNetwork> transition_model;
    std::unique_ptr<DynamicNeuralNetwork> reward_model;
    std::unique_ptr<DynamicNeuralNetwork> observation_model;
    int state_dim_;
    int action_dim_;
    
public:
    WorldModel(int state_dim, int action_dim, int latent_dim = 32,
               const std::vector<int>& hidden_layers = {128, 64});
    
    std::pair<State, float> predict(const State& state, const Action& action);
    void train_step(const std::vector<Transition>& batch);
    float imagine_and_evaluate(std::unique_ptr<RLAgent>& agent, const State& state, 
                              int horizon = 10);
};

// Imagination Agent
class ImaginationAgent : public RLAgent {
private:
    std::unique_ptr<WorldModel> world_model;
    std::unique_ptr<RLAgent> base_agent;
    int imagination_horizon;
    float imagination_weight;
    
public:
    ImaginationAgent(std::unique_ptr<RLAgent> agent, std::unique_ptr<WorldModel> world_model,
                     int imagination_horizon = 10, float imagination_weight = 0.5f);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
};

// Multi-Agent RL - MADDPG
class MADDPGAgent {
private:
    std::unique_ptr<DynamicNeuralNetwork> actor_network;
    std::unique_ptr<DynamicNeuralNetwork> critic_network;
    std::unique_ptr<DynamicNeuralNetwork> target_actor;
    std::unique_ptr<DynamicNeuralNetwork> target_critic;
    std::mt19937 rng;
    int agent_id;
    int num_agents;
    int state_dim_;
    int action_dim_;
    
public:
    MADDPGAgent(int agent_id, int num_agents, int state_dim, int action_dim,
                const std::vector<int>& hidden_layers = {128, 64},
                float lr = 0.001, float gamma = 0.99);
    
    Action select_action(const State& state, bool explore = true);
    void train_step(const std::vector<std::vector<Transition>>& joint_batch);
    void update_target_network();
};

// QMIX Agent
class QMIXAgent {
private:
    std::unique_ptr<DynamicNeuralNetwork> agent_network;
    std::unique_ptr<DynamicNeuralNetwork> mixing_network;
    std::mt19937 rng;
    int num_agents;
    int agent_obs_dim;
    int agent_action_dim;
    
public:
    QMIXAgent(int num_agents, int agent_obs_dim, int agent_action_dim,
              const std::vector<int>& hidden_layers = {128, 64},
              float lr = 0.001, float gamma = 0.99);
    
    std::vector<Action> select_actions(const std::vector<State>& states, bool explore = true);
    void train_step(const std::vector<std::vector<Transition>>& joint_batch);
    float compute_qmix_value(const std::vector<State>& states, const std::vector<Action>& actions);
};

// VDN Agent
class VDNAgent {
private:
    std::vector<std::unique_ptr<DynamicNeuralNetwork>> agent_networks;
    std::mt19937 rng;
    int num_agents;
    int agent_obs_dim;
    int agent_action_dim;
    
public:
    VDNAgent(int num_agents, int agent_obs_dim, int agent_action_dim,
             const std::vector<int>& hidden_layers = {128, 64},
             float lr = 0.001, float gamma = 0.99);
    
    std::vector<Action> select_actions(const std::vector<State>& states, bool explore = true);
    void train_step(const std::vector<std::vector<Transition>>& joint_batch);
    float compute_vdn_value(const std::vector<State>& states, const std::vector<Action>& actions);
};

// Hierarchical RL - Options
class Option {
private:
    std::unique_ptr<RLAgent> policy;
    std::function<bool(const State&)> termination_condition;
    int option_id;
    
public:
    Option(int option_id, std::unique_ptr<RLAgent> policy,
           std::function<bool(const State&)> termination);
    
    bool is_terminated(const State& state);
    Action select_action(const State& state, bool explore = true);
    void train_step(const std::vector<Transition>& batch);
};

// Hierarchical Actor-Critic (HAC)
class HACAgent : public RLAgent {
private:
    std::vector<std::unique_ptr<Option>> options;
    std::unique_ptr<DynamicNeuralNetwork> high_level_policy;
    int current_option;
    int num_options;
    
public:
    HACAgent(int state_dim, int action_dim, int num_options = 4,
             const std::vector<int>& hidden_layers = {128, 64},
             float lr = 0.001, float gamma = 0.99);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
    void add_option(std::unique_ptr<Option> option);
};

// FeUdal Network (FuN)
class FeUdalNetwork {
private:
    std::unique_ptr<DynamicNeuralNetwork> manager_network;
    std::unique_ptr<DynamicNeuralNetwork> worker_network;
    int goal_dim;
    int horizon;
    
public:
    FeUdalNetwork(int state_dim, int action_dim, int goal_dim = 16, int horizon = 10,
                  const std::vector<int>& hidden_layers = {128, 64},
                  float lr = 0.001, float gamma = 0.99);
    
    std::vector<float> generate_goal(const State& state);
    Action select_action(const State& state, const std::vector<float>& goal, int t);
    void train_step(const std::vector<Transition>& batch);
};

// Offline RL - Conservative Q-Learning (CQL)
class CQLAgent : public DQN {
private:
    float alpha;  // Conservative coefficient
    
public:
    CQLAgent(int state_dim, int action_dim, const std::vector<int>& hidden_layers = {128, 64},
             float lr = 0.001, float gamma = 0.99, float alpha = 5.0f);
    
    void train_step(const std::vector<Transition>& batch) override;
    float compute_cql_loss(const std::vector<Transition>& batch);
};

// Batch Constrained Q-learning (BCQ)
class BCQAgent : public DQN {
private:
    std::unique_ptr<DynamicNeuralNetwork> perturbation_network;
    float threshold;
    
public:
    BCQAgent(int state_dim, int action_dim, const std::vector<int>& hidden_layers = {128, 64},
             float lr = 0.001, float gamma = 0.99, float threshold = 0.3f);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
    
    Action generate_perturbed_action(const State& state, const Action& action);
};

// Meta-RL - MAML
class MAMLAgent : public RLAgent {
private:
    std::unique_ptr<RLAgent> base_agent;
    std::vector<std::vector<Transition>> task_data;
    int inner_steps;
    float inner_lr;
    
public:
    MAMLAgent(std::unique_ptr<RLAgent> base_agent, int inner_steps = 5, float inner_lr = 0.01f);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
    void adapt_to_task(const std::vector<Transition>& task_batch);
    void meta_update(const std::vector<std::vector<Transition>>& tasks);
};

// Reptile Algorithm
class ReptileAgent : public RLAgent {
private:
    std::unique_ptr<RLAgent> base_agent;
    int inner_steps;
    float epsilon;  // Interpolation factor
    
public:
    ReptileAgent(std::unique_ptr<RLAgent> base_agent, int inner_steps = 5, float epsilon = 1.0f);
    
    Action select_action(const State& state, bool explore = true) override;
    void train_step(const std::vector<Transition>& batch) override;
    void meta_update(const std::vector<std::vector<Transition>>& tasks);
};

// Training utilities
class RLTrainer {
private:
    std::unique_ptr<RLAgent> agent;
    std::unique_ptr<Environment> environment;
    std::unique_ptr<ReplayBuffer> replay_buffer;
    
public:
    RLTrainer(std::unique_ptr<RLAgent> agent, std::unique_ptr<Environment> env,
              size_t buffer_size = 100000);
    
    void train(int num_episodes, int max_steps_per_episode, int batch_size = 32,
               int update_frequency = 1, int target_update_frequency = 100);
    
    std::vector<float> evaluate(int num_episodes = 10);
    void save_model(const std::string& filepath);
    void load_model(const std::string& filepath);
};

// Utility functions
namespace RLUtils {
    float compute_discounted_returns(const std::vector<float>& rewards, float gamma);
    std::vector<float> compute_discounted_returns_vector(const std::vector<float>& rewards, float gamma);
    std::vector<float> compute_advantages(const std::vector<float>& rewards, 
                                         const std::vector<float>& values, float gamma, float lambda = 0.95f);
    State normalize_state(const State& state, const std::vector<float>& mean, const std::vector<float>& std);
    std::pair<std::vector<float>, std::vector<float>> compute_running_stats(const std::vector<State>& states);
    Action epsilon_greedy_action(const std::vector<float>& q_values, float epsilon, int action_dim);
    Action categorical_sample(const std::vector<float>& probabilities);
    std::vector<float> softmax(const std::vector<float>& logits);
    float kl_divergence(const std::vector<float>& p, const std::vector<float>& q);
}

} // namespace TinyML

#endif // REINFORCEMENTLEARNING_H
