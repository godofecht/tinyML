#include "ReinforcementLearning.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace TinyML {

namespace {
std::vector<float> encode_action_vector(const Action& action, int action_dim) {
    if (!action.continuous.empty()) {
        return action.continuous;
    }
    std::vector<float> encoded(action_dim, 0.0f);
    if (action.action_id >= 0 && action.action_id < action_dim) {
        encoded[static_cast<size_t>(action.action_id)] = 1.0f;
    }
    return encoded;
}
} // namespace

// DQN Implementation
DQN::DQN(int state_dim, int action_dim, const std::vector<int>& hidden_layers,
         float lr, float gamma, float epsilon_start, float epsilon_decay, float epsilon_min)
    : RLAgent(state_dim, action_dim, lr, gamma), epsilon(epsilon_start),
      epsilon_decay(epsilon_decay), epsilon_min(epsilon_min),
      update_frequency(1), steps_since_update(0) {
    
    // Create Q-network
    q_network = std::make_unique<DynamicNeuralNetwork>();
    q_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        q_network->addLayer(hidden);
    }
    q_network->addLayer(action_dim);
    
    // Create target network
    target_network = std::make_unique<DynamicNeuralNetwork>();
    target_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        target_network->addLayer(hidden);
    }
    target_network->addLayer(action_dim);
    
    // Initialize target network
    update_target_network();
}

Action DQN::select_action(const State& state, bool explore) {
    std::vector<float> q_values = get_q_values(state);
    
    if (explore && rng() % 10000 < epsilon * 10000) {
        // Random action
        std::uniform_int_distribution<int> dist(0, action_dim - 1);
        return Action(dist(rng));
    } else {
        // Greedy action
        auto max_it = std::max_element(q_values.begin(), q_values.end());
        int action = std::distance(q_values.begin(), max_it);
        return Action(action);
    }
}

void DQN::train_step(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    
    // Compute target Q-values
    std::vector<float> targets(batch.size() * action_dim);
    std::vector<float> inputs(batch.size() * state_dim);
    
    for (size_t i = 0; i < batch.size(); ++i) {
        const auto& transition = batch[i];
        
        // Copy state to inputs
        for (int j = 0; j < state_dim; ++j) {
            inputs[i * state_dim + j] = transition.state.features[j];
        }
        
        // Compute target
        std::vector<float> next_q_values = get_q_values(transition.next_state);
        float max_next_q = *std::max_element(next_q_values.begin(), next_q_values.end());
        float target = transition.reward + gamma * (transition.done ? 0.0f : max_next_q);
        
        // Set targets (only for the taken action)
        std::vector<float> current_q = get_q_values(transition.state);
        for (int j = 0; j < action_dim; ++j) {
            targets[i * action_dim + j] = (j == transition.action.action_id) ? target : current_q[j];
        }
    }
    
    // Train network (simplified - in practice would use proper backprop)
    for (size_t i = 0; i < batch.size(); ++i) {
        std::vector<float> input(inputs.begin() + i * state_dim, 
                                inputs.begin() + (i + 1) * state_dim);
        std::vector<float> target(targets.begin() + i * action_dim,
                                  targets.begin() + (i + 1) * action_dim);
        
        // Forward pass
        std::vector<float> output = q_network->forward(input);
        
        // Compute loss and update weights (simplified)
        for (int j = 0; j < action_dim; ++j) {
            float error = target[j] - output[j];
            // In practice, this would be proper gradient descent
        }
    }
    
    // Decay epsilon
    epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
    
    // Update target network periodically
    steps_since_update++;
    if (steps_since_update >= update_frequency) {
        update_target_network();
        steps_since_update = 0;
    }
}

void DQN::update_target_network() {
    if (!q_network || !target_network) {
        return;
    }

    // Placeholder: DynamicNeuralNetwork does not expose weights for copying yet.
    // In a full implementation, this would sync weights from q_network to target_network.
}

// Model-Based RL - World Model
WorldModel::WorldModel(int state_dim, int action_dim, int latent_dim,
                       const std::vector<int>& hidden_layers) {
    state_dim_ = state_dim;
    action_dim_ = action_dim;
    
    // Transition model: (state, action) -> next_state
    transition_model = std::make_unique<DynamicNeuralNetwork>();
    transition_model->addLayer(state_dim + action_dim);
    for (int hidden : hidden_layers) {
        transition_model->addLayer(hidden);
    }
    transition_model->addLayer(state_dim);
    
    // Reward model: (state, action) -> reward
    reward_model = std::make_unique<DynamicNeuralNetwork>();
    reward_model->addLayer(state_dim + action_dim);
    for (int hidden : hidden_layers) {
        reward_model->addLayer(hidden);
    }
    reward_model->addLayer(1);
    
    // Observation model: latent -> state
    observation_model = std::make_unique<DynamicNeuralNetwork>();
    observation_model->addLayer(latent_dim);
    for (int hidden : hidden_layers) {
        observation_model->addLayer(hidden);
    }
    observation_model->addLayer(state_dim);
}

std::pair<State, float> WorldModel::predict(const State& state, const Action& action) {
    std::vector<float> sa(state.features.begin(), state.features.end());
    auto action_vec = encode_action_vector(action, action_dim_);
    sa.insert(sa.end(), action_vec.begin(), action_vec.end());
    
    std::vector<float> next_state_features = transition_model->forward(sa);
    std::vector<float> reward_output = reward_model->forward(sa);
    
    return {State(next_state_features), reward_output[0]};
}

void WorldModel::train_step(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    
    for (const auto& transition : batch) {
        // Train transition model
        std::vector<float> sa(transition.state.features.begin(), transition.state.features.end());
        auto action_vec = encode_action_vector(transition.action, action_dim_);
        sa.insert(sa.end(), action_vec.begin(), action_vec.end());
        
        std::vector<float> predicted_next_state = transition_model->forward(sa);
        float transition_loss = 0.0f;
        for (size_t i = 0; i < transition.next_state.features.size(); ++i) {
            float error = predicted_next_state[i] - transition.next_state.features[i];
            transition_loss += error * error;
        }
        
        // Train reward model
        std::vector<float> predicted_reward = reward_model->forward(sa);
        float reward_loss = (predicted_reward[0] - transition.reward) * (predicted_reward[0] - transition.reward);
        
        // Update models (simplified)
    }
}

float WorldModel::imagine_and_evaluate(std::unique_ptr<RLAgent>& agent, const State& state, int horizon) {
    float total_reward = 0.0f;
    State current_state = state;
    float discount = 1.0f;
    
    for (int t = 0; t < horizon; ++t) {
        Action action = agent->select_action(current_state, false);
        auto [next_state, reward] = predict(current_state, action);
        
        total_reward += discount * reward;
        discount *= 0.99f;  // Gamma
        
        current_state = next_state;
    }
    
    return total_reward;
}

// Imagination Agent
ImaginationAgent::ImaginationAgent(std::unique_ptr<RLAgent> agent, std::unique_ptr<WorldModel> world_model,
                                 int imagination_horizon, float imagination_weight)
    : RLAgent(agent->get_state_dim(), agent->get_action_dim(),
              agent->get_learning_rate(), agent->get_gamma()),
      world_model(std::move(world_model)), base_agent(std::move(agent)),
      imagination_horizon(imagination_horizon), imagination_weight(imagination_weight) {}

Action ImaginationAgent::select_action(const State& state, bool explore) {
    // Combine real and imagined Q-values
    float imagined_value = world_model->imagine_and_evaluate(base_agent, state, imagination_horizon);
    
    // Use imagination to bias action selection (simplified)
    Action action = base_agent->select_action(state, explore);
    
    return action;
}

void ImaginationAgent::train_step(const std::vector<Transition>& batch) {
    // Train world model
    world_model->train_step(batch);
    
    // Train base agent with imagined experience
    std::vector<Transition> imagined_batch;
    for (const auto& transition : batch) {
        // Generate imagined trajectories
        State current_state = transition.state;
        for (int t = 0; t < imagination_horizon; ++t) {
            Action action = base_agent->select_action(current_state, false);
            auto [next_state, reward] = world_model->predict(current_state, action);
            
            imagined_batch.emplace_back(current_state, action, reward * imagination_weight, next_state, false);
            current_state = next_state;
        }
    }
    
    // Train base agent with combined real and imagined experience
    std::vector<Transition> combined_batch = batch;
    combined_batch.insert(combined_batch.end(), imagined_batch.begin(), imagined_batch.end());
    base_agent->train_step(combined_batch);
}

// MADDPG Agent
MADDPGAgent::MADDPGAgent(int agent_id, int num_agents, int state_dim, int action_dim,
                         const std::vector<int>& hidden_layers, float lr, float gamma)
    : rng(std::random_device{}()), agent_id(agent_id), num_agents(num_agents),
      state_dim_(state_dim), action_dim_(action_dim) {
    
    // Actor network
    actor_network = std::make_unique<DynamicNeuralNetwork>();
    actor_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        actor_network->addLayer(hidden);
    }
    actor_network->addLayer(action_dim);
    
    // Critic network (takes all agents' states and actions)
    critic_network = std::make_unique<DynamicNeuralNetwork>();
    critic_network->addLayer(state_dim * num_agents + action_dim * num_agents);
    for (int hidden : hidden_layers) {
        critic_network->addLayer(hidden);
    }
    critic_network->addLayer(1);
    
    // Target networks
    target_actor = std::make_unique<DynamicNeuralNetwork>();
    target_actor->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        target_actor->addLayer(hidden);
    }
    target_actor->addLayer(action_dim);
    
    target_critic = std::make_unique<DynamicNeuralNetwork>();
    target_critic->addLayer(state_dim * num_agents + action_dim * num_agents);
    for (int hidden : hidden_layers) {
        target_critic->addLayer(hidden);
    }
    target_critic->addLayer(1);
}

Action MADDPGAgent::select_action(const State& state, bool explore) {
    std::vector<float> action_values = actor_network->forward(state.features);
    
    if (explore) {
        std::normal_distribution<float> noise(0.0f, 0.1f);
        for (float& val : action_values) {
            val += noise(rng);
        }
    }
    
    return Action(action_values);
}

void MADDPGAgent::train_step(const std::vector<std::vector<Transition>>& joint_batch) {
    if (joint_batch.empty() || joint_batch[agent_id].empty()) return;
    
    for (const auto& transition : joint_batch[agent_id]) {
        // Combine all agents' states and actions for critic
        std::vector<float> joint_input;
        for (int i = 0; i < num_agents; ++i) {
            // Add states
            joint_input.insert(joint_input.end(), transition.state.features.begin(), transition.state.features.end());
        }
        for (int i = 0; i < num_agents; ++i) {
            // Add actions
            auto action_vec = encode_action_vector(transition.action, action_dim_);
            joint_input.insert(joint_input.end(), action_vec.begin(), action_vec.end());
        }
        
        float q_value = critic_network->forward(joint_input)[0];
        
        // Compute target Q-value
        float target = transition.reward;
        if (!transition.done) {
            std::vector<float> next_joint_input;
            for (int i = 0; i < num_agents; ++i) {
                next_joint_input.insert(next_joint_input.end(), transition.next_state.features.begin(), transition.next_state.features.end());
            }
            // Use target actors for next actions
            std::vector<float> next_actions = target_actor->forward(transition.next_state.features);
            for (int i = 0; i < num_agents; ++i) {
                next_joint_input.insert(next_joint_input.end(), next_actions.begin(), next_actions.end());
            }
            
            target += 0.99f * target_critic->forward(next_joint_input)[0];
        }
        
        float critic_loss = (q_value - target) * (q_value - target);
        
        // Update actor
        std::vector<float> actor_output = actor_network->forward(transition.state.features);
        float actor_loss = -critic_network->forward(joint_input)[0];
        
        // Update networks (simplified)
    }
}

void MADDPGAgent::update_target_network() {
    // Soft update target networks
}

// QMIX Agent
QMIXAgent::QMIXAgent(int num_agents, int agent_obs_dim, int agent_action_dim,
                      const std::vector<int>& hidden_layers, float lr, float gamma)
    : rng(std::random_device{}()), num_agents(num_agents), agent_obs_dim(agent_obs_dim),
      agent_action_dim(agent_action_dim) {
    
    agent_network = std::make_unique<DynamicNeuralNetwork>();
    agent_network->addLayer(agent_obs_dim);
    for (int hidden : hidden_layers) {
        agent_network->addLayer(hidden);
    }
    agent_network->addLayer(agent_action_dim);
    
    mixing_network = std::make_unique<DynamicNeuralNetwork>();
    mixing_network->addLayer(num_agents * agent_action_dim);
    for (int hidden : hidden_layers) {
        mixing_network->addLayer(hidden);
    }
    mixing_network->addLayer(1);
}

std::vector<Action> QMIXAgent::select_actions(const std::vector<State>& states, bool explore) {
    std::vector<Action> actions;

    auto normalize_input = [](const std::vector<float>& input, size_t expected) {
        if (input.size() == expected) {
            return input;
        }
        std::vector<float> normalized(expected, 0.0f);
        size_t count = std::min(input.size(), expected);
        std::copy_n(input.begin(), count, normalized.begin());
        return normalized;
    };
    
    for (const auto& state : states) {
        std::vector<float> normalized_state = normalize_input(state.features, agent_obs_dim);
        std::vector<float> q_values;
        try {
            q_values = agent_network->forward(normalized_state);
        } catch (const std::invalid_argument&) {
            actions.emplace_back(0);
            continue;
        }
        
        if (explore) {
            std::uniform_int_distribution<int> dist(0, agent_action_dim - 1);
            actions.emplace_back(dist(rng));
        } else {
            auto max_it = std::max_element(q_values.begin(), q_values.end());
            int action = std::distance(q_values.begin(), max_it);
            actions.emplace_back(action);
        }
    }
    
    return actions;
}

void QMIXAgent::train_step(const std::vector<std::vector<Transition>>& joint_batch) {
    if (joint_batch.empty()) return;

    auto normalize_input = [](const std::vector<float>& input, size_t expected) {
        if (input.size() == expected) {
            return input;
        }
        std::vector<float> normalized(expected, 0.0f);
        size_t count = std::min(input.size(), expected);
        std::copy_n(input.begin(), count, normalized.begin());
        return normalized;
    };

    try {
        // Train individual agents
        for (int agent_idx = 0; agent_idx < num_agents; ++agent_idx) {
            for (const auto& transition : joint_batch[agent_idx]) {
                std::vector<float> normalized_state = normalize_input(transition.state.features, agent_obs_dim);
                std::vector<float> q_values = agent_network->forward(normalized_state);
                float q_value = q_values[transition.action.action_id];
                
                float target = transition.reward;
                if (!transition.done) {
                    std::vector<float> normalized_next = normalize_input(transition.next_state.features, agent_obs_dim);
                    std::vector<float> next_q_values = agent_network->forward(normalized_next);
                    float max_next_q = *std::max_element(next_q_values.begin(), next_q_values.end());
                    target += 0.99f * max_next_q;
                }
                
                float loss = (q_value - target) * (q_value - target);
                // Update agent network (simplified)
            }
        }
        
        // Train mixing network
        std::vector<float> joint_q_values(num_agents * agent_action_dim, 0.0f);
        for (int agent_idx = 0; agent_idx < num_agents; ++agent_idx) {
            if (!joint_batch[agent_idx].empty()) {
                const auto& transition = joint_batch[agent_idx][0];
                std::vector<float> normalized_state = normalize_input(transition.state.features, agent_obs_dim);
                std::vector<float> q_values = agent_network->forward(normalized_state);
                size_t offset = static_cast<size_t>(agent_idx) * agent_action_dim;
                for (int i = 0; i < agent_action_dim && i < static_cast<int>(q_values.size()); ++i) {
                    joint_q_values[offset + static_cast<size_t>(i)] = q_values[static_cast<size_t>(i)];
                }
            }
        }

        if (joint_q_values.size() != static_cast<size_t>(num_agents * agent_action_dim)) {
            joint_q_values.resize(static_cast<size_t>(num_agents * agent_action_dim), 0.0f);
        }
        
        float joint_q = mixing_network->forward(joint_q_values)[0];
        float joint_target = 0.0f;
        for (const auto& agent_batch : joint_batch) {
            if (!agent_batch.empty()) {
                joint_target += agent_batch[0].reward;
            }
        }
        
        float mixing_loss = (joint_q - joint_target) * (joint_q - joint_target);
        // Update mixing network (simplified)
    } catch (const std::invalid_argument&) {
        return;
    }
}

float QMIXAgent::compute_qmix_value(const std::vector<State>& states, const std::vector<Action>& actions) {
    std::vector<float> joint_q_values;
    
    for (size_t i = 0; i < states.size(); ++i) {
        std::vector<float> q_values = agent_network->forward(states[i].features);
        joint_q_values.push_back(q_values[actions[i].action_id]);
    }
    
    return mixing_network->forward(joint_q_values)[0];
}

// VDN Agent
VDNAgent::VDNAgent(int num_agents, int agent_obs_dim, int agent_action_dim,
                   const std::vector<int>& hidden_layers, float lr, float gamma)
    : rng(std::random_device{}()), num_agents(num_agents), agent_obs_dim(agent_obs_dim),
      agent_action_dim(agent_action_dim) {
    
    for (int i = 0; i < num_agents; ++i) {
        auto network = std::make_unique<DynamicNeuralNetwork>();
        network->addLayer(agent_obs_dim);
        for (int hidden : hidden_layers) {
            network->addLayer(hidden);
        }
        network->addLayer(agent_action_dim);
        agent_networks.push_back(std::move(network));
    }
}

std::vector<Action> VDNAgent::select_actions(const std::vector<State>& states, bool explore) {
    std::vector<Action> actions;
    
    for (size_t i = 0; i < states.size() && i < agent_networks.size(); ++i) {
        std::vector<float> q_values = agent_networks[i]->forward(states[i].features);
        
        if (explore) {
            std::uniform_int_distribution<int> dist(0, agent_action_dim - 1);
            actions.emplace_back(dist(rng));
        } else {
            auto max_it = std::max_element(q_values.begin(), q_values.end());
            int action = std::distance(q_values.begin(), max_it);
            actions.emplace_back(action);
        }
    }
    
    return actions;
}

void VDNAgent::train_step(const std::vector<std::vector<Transition>>& joint_batch) {
    for (int agent_idx = 0; agent_idx < num_agents; ++agent_idx) {
        if (agent_idx >= joint_batch.size()) continue;
        
        for (const auto& transition : joint_batch[agent_idx]) {
            std::vector<float> q_values = agent_networks[agent_idx]->forward(transition.state.features);
            float q_value = q_values[transition.action.action_id];
            
            float target = transition.reward;
            if (!transition.done) {
                std::vector<float> next_q_values = agent_networks[agent_idx]->forward(transition.next_state.features);
                float max_next_q = *std::max_element(next_q_values.begin(), next_q_values.end());
                target += 0.99f * max_next_q;
            }
            
            float loss = (q_value - target) * (q_value - target);
            // Update agent network (simplified)
        }
    }
}

float VDNAgent::compute_vdn_value(const std::vector<State>& states, const std::vector<Action>& actions) {
    float total_q = 0.0f;
    
    for (size_t i = 0; i < states.size() && i < agent_networks.size(); ++i) {
        std::vector<float> q_values = agent_networks[i]->forward(states[i].features);
        total_q += q_values[actions[i].action_id];
    }
    
    return total_q;
}

std::vector<float> DQN::get_q_values(const State& state) {
    return q_network->forward(state.features);
}

// Double DQN Implementation
DoubleDQN::DoubleDQN(int state_dim, int action_dim, const std::vector<int>& hidden_layers,
                     float lr, float gamma, float epsilon_start, float epsilon_decay, float epsilon_min)
    : DQN(state_dim, action_dim, hidden_layers, lr, gamma, epsilon_start, epsilon_decay, epsilon_min) {}

void DoubleDQN::train_step(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    
    // Double DQN uses main network to select actions and target network to evaluate
    for (const auto& transition : batch) {
        std::vector<float> next_q_main = get_q_values(transition.next_state);
        auto max_it = std::max_element(next_q_main.begin(), next_q_main.end());
        int best_action = std::distance(next_q_main.begin(), max_it);
        
        std::vector<float> next_q_target = target_network->forward(transition.next_state.features);
        float target = transition.reward + gamma * (transition.done ? 0.0f : next_q_target[best_action]);
        
        // Update Q-value for the taken action
        // (simplified implementation)
    }
    
    epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
}

// Dueling DQN Implementation
DuelingDQN::DuelingDQN(int state_dim, int action_dim, const std::vector<int>& hidden_layers,
                       float lr, float gamma, float epsilon_start, float epsilon_decay, float epsilon_min)
    : DQN(state_dim, action_dim, hidden_layers, lr, gamma, epsilon_start, epsilon_decay, epsilon_min) {
    
    // Create value stream
    value_stream = std::make_unique<DynamicNeuralNetwork>();
    value_stream->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        value_stream->addLayer(hidden);
    }
    value_stream->addLayer(1);  // Single value output
    
    // Create advantage stream
    advantage_stream = std::make_unique<DynamicNeuralNetwork>();
    advantage_stream->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        advantage_stream->addLayer(hidden);
    }
    advantage_stream->addLayer(action_dim);  // Advantage for each action
}

std::vector<float> DuelingDQN::get_q_values(const State& state) {
    std::vector<float> value = value_stream->forward(state.features);
    std::vector<float> advantages = advantage_stream->forward(state.features);
    
    std::vector<float> q_values(action_dim);
    float mean_advantage = std::accumulate(advantages.begin(), advantages.end(), 0.0f) / action_dim;
    
    for (int i = 0; i < action_dim; ++i) {
        q_values[i] = value[0] + advantages[i] - mean_advantage;
    }
    
    return q_values;
}

// REINFORCE Implementation
REINFORCE::REINFORCE(int state_dim, int action_dim, const std::vector<int>& hidden_layers,
                     float lr, float gamma)
    : RLAgent(state_dim, action_dim, lr, gamma) {
    
    policy_network = std::make_unique<DynamicNeuralNetwork>();
    policy_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        policy_network->addLayer(hidden);
    }
    policy_network->addLayer(action_dim);  // Policy logits
}

Action REINFORCE::select_action(const State& state, bool explore) {
    std::vector<float> logits = policy_network->forward(state.features);
    std::vector<float> probs = RLUtils::softmax(logits);
    
    if (explore) {
        return RLUtils::categorical_sample(probs);
    } else {
        auto max_it = std::max_element(probs.begin(), probs.end());
        int action = std::distance(probs.begin(), max_it);
        return Action(action);
    }
}

void REINFORCE::train_step(const std::vector<Transition>& batch) {
    // REINFORCE updates based on complete episodes
    // This is a simplified version
    if (rewards.empty()) return;
    
    // Compute discounted returns
    std::vector<float> returns = RLUtils::compute_discounted_returns_vector(rewards, gamma);
    
    // Update policy
    for (size_t t = 0; t < log_probs.size(); ++t) {
        float loss = -log_probs[t] * returns[t];
        // Update policy network (simplified)
    }
    
    // Clear episode data
    log_probs.clear();
    rewards.clear();
}

void REINFORCE::update_policy() {
    // Perform policy update with collected episode data
    train_step({});
}

// A2C Implementation
A2C::A2C(int state_dim, int action_dim, const std::vector<int>& hidden_layers,
         float lr, float gamma, float entropy_coeff)
    : RLAgent(state_dim, action_dim, lr, gamma), entropy_coefficient(entropy_coeff) {
    
    // Actor network
    actor_network = std::make_unique<DynamicNeuralNetwork>();
    actor_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        actor_network->addLayer(hidden);
    }
    actor_network->addLayer(action_dim);
    
    // Critic network
    critic_network = std::make_unique<DynamicNeuralNetwork>();
    critic_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        critic_network->addLayer(hidden);
    }
    critic_network->addLayer(1);  // State value
}

Action A2C::select_action(const State& state, bool explore) {
    std::vector<float> logits = actor_network->forward(state.features);
    std::vector<float> probs = RLUtils::softmax(logits);
    
    if (explore) {
        return RLUtils::categorical_sample(probs);
    } else {
        auto max_it = std::max_element(probs.begin(), probs.end());
        int action = std::distance(probs.begin(), max_it);
        return Action(action);
    }
}

void A2C::train_step(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    
    for (const auto& transition : batch) {
        // Get value estimates
        float value = critic_network->forward(transition.state.features)[0];
        float next_value = transition.done ? 0.0f : 
                          critic_network->forward(transition.next_state.features)[0];
        
        // Compute advantage
        float advantage = transition.reward + gamma * next_value - value;
        
        // Update actor (policy gradient)
        std::vector<float> logits = actor_network->forward(transition.state.features);
        std::vector<float> probs = RLUtils::softmax(logits);
        
        // Actor loss: -log_prob * advantage
        float actor_loss = -std::log(probs[transition.action.action_id]) * advantage;
        
        // Add entropy bonus
        float entropy = -std::inner_product(probs.begin(), probs.end(), probs.begin(), 0.0f,
                                          [](float a, float b) { return a + b; },
                                          [](float p, float) { return p * std::log(p + 1e-8f); });
        actor_loss -= entropy_coefficient * entropy;
        
        // Update critic (value function)
        float critic_loss = advantage * advantage;
        
        // Update networks (simplified)
    }
}

std::pair<float, float> A2C::evaluate_state(const State& state) {
    std::vector<float> logits = actor_network->forward(state.features);
    std::vector<float> probs = RLUtils::softmax(logits);
    float value = critic_network->forward(state.features)[0];
    
    return {probs[0], value};  // Simplified return
}

// A3C Implementation
A3C::A3C(int state_dim, int action_dim, int num_workers,
         const std::vector<int>& hidden_layers, float lr, float gamma, float entropy_coeff)
    : A2C(state_dim, action_dim, hidden_layers, lr, gamma, entropy_coeff),
      num_workers(num_workers) {
    
    // Create worker agents
    for (int i = 0; i < num_workers; ++i) {
        workers.push_back(std::make_unique<A2C>(state_dim, action_dim, hidden_layers, lr, gamma, entropy_coeff));
    }
}

void A3C::train_step(const std::vector<Transition>& batch) {
    // A3C would implement parallel training
    // This is a simplified version that just calls A2C training
    A2C::train_step(batch);
}

void A3C::train_parallel(Environment& env, int max_episodes) {
    // Parallel training implementation would go here
    // For now, just do sequential training
    for (int episode = 0; episode < max_episodes; ++episode) {
        State state = env.reset();
        float total_reward = 0.0f;
        
        while (!state.terminal) {
            Action action = select_action(state, true);
            auto [next_state, reward] = env.step(action);
            
            std::vector<Transition> batch = {Transition(state, action, reward, next_state, next_state.terminal)};
            train_step(batch);
            
            state = next_state;
            total_reward += reward;
        }
        
        if (episode % 100 == 0) {
            std::cout << "Episode " << episode << ", Total Reward: " << total_reward << std::endl;
        }
    }
}

// PPO Implementation
PPO::PPO(int state_dim, int action_dim, const std::vector<int>& hidden_layers,
         float lr, float gamma, float clip_epsilon, float entropy_coeff, int ppo_epochs, int mini_batch_size)
    : RLAgent(state_dim, action_dim, lr, gamma), clip_epsilon(clip_epsilon),
      entropy_coefficient(entropy_coeff), ppo_epochs(ppo_epochs), mini_batch_size(mini_batch_size) {
    
    actor_network = std::make_unique<DynamicNeuralNetwork>();
    actor_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        actor_network->addLayer(hidden);
    }
    actor_network->addLayer(action_dim);
    
    critic_network = std::make_unique<DynamicNeuralNetwork>();
    critic_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        critic_network->addLayer(hidden);
    }
    critic_network->addLayer(1);
    
    old_actor_network = std::make_unique<DynamicNeuralNetwork>();
    old_actor_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        old_actor_network->addLayer(hidden);
    }
    old_actor_network->addLayer(action_dim);
    
    update_old_policy();
}

Action PPO::select_action(const State& state, bool explore) {
    std::vector<float> logits = actor_network->forward(state.features);
    std::vector<float> probs = RLUtils::softmax(logits);
    
    if (explore) {
        return RLUtils::categorical_sample(probs);
    } else {
        auto max_it = std::max_element(probs.begin(), probs.end());
        int action = std::distance(probs.begin(), max_it);
        return Action(action);
    }
}

void PPO::train_step(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    
    for (int epoch = 0; epoch < ppo_epochs; ++epoch) {
        // Shuffle batch
        std::vector<Transition> shuffled_batch = batch;
        std::shuffle(shuffled_batch.begin(), shuffled_batch.end(), rng);
        
        // Process mini-batches
        for (size_t i = 0; i < shuffled_batch.size(); i += mini_batch_size) {
            size_t end = std::min(i + mini_batch_size, shuffled_batch.size());
            std::vector<Transition> mini_batch(shuffled_batch.begin() + i, shuffled_batch.begin() + end);
            
            float loss = compute_ppo_loss(mini_batch);
            
            // Update networks (simplified)
        }
    }
    
    update_old_policy();
}

void PPO::update_old_policy() {
    // Copy current actor weights to old actor
    // In practice, this would copy actual parameters
}

float PPO::compute_ppo_loss(const std::vector<Transition>& batch) {
    float total_loss = 0.0f;
    
    for (const auto& transition : batch) {
        // Get current and old policy probabilities
        std::vector<float> current_logits = actor_network->forward(transition.state.features);
        std::vector<float> old_logits = old_actor_network->forward(transition.state.features);
        std::vector<float> current_probs = RLUtils::softmax(current_logits);
        std::vector<float> old_probs = RLUtils::softmax(old_logits);
        
        // Compute ratio
        float ratio = current_probs[transition.action.action_id] / 
                     (old_probs[transition.action.action_id] + 1e-8f);
        
        // Compute advantage
        float value = critic_network->forward(transition.state.features)[0];
        float next_value = transition.done ? 0.0f : 
                          critic_network->forward(transition.next_state.features)[0];
        float advantage = transition.reward + gamma * next_value - value;
        
        // PPO clipped loss
        float clipped_ratio = std::clamp(ratio, 1.0f - clip_epsilon, 1.0f + clip_epsilon);
        float policy_loss = -std::min(ratio * advantage, clipped_ratio * advantage);
        
        // Value loss
        float value_loss = advantage * advantage;
        
        // Entropy bonus
        float entropy = -std::inner_product(current_probs.begin(), current_probs.end(), 
                                          current_probs.begin(), 0.0f,
                                          [](float a, float b) { return a + b; },
                                          [](float p, float) { return p * std::log(p + 1e-8f); });
        
        total_loss += policy_loss + 0.5f * value_loss - entropy_coefficient * entropy;
    }
    
    return total_loss / batch.size();
}

// TRPO Implementation
TRPO::TRPO(int state_dim, int action_dim, const std::vector<int>& hidden_layers,
           float lr, float gamma, float delta, int cg_iters)
    : RLAgent(state_dim, action_dim, lr, gamma), delta(delta), cg_iterations(cg_iters) {
    
    actor_network = std::make_unique<DynamicNeuralNetwork>();
    actor_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        actor_network->addLayer(hidden);
    }
    actor_network->addLayer(action_dim);
    
    critic_network = std::make_unique<DynamicNeuralNetwork>();
    critic_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        critic_network->addLayer(hidden);
    }
    critic_network->addLayer(1);
}

Action TRPO::select_action(const State& state, bool explore) {
    std::vector<float> logits = actor_network->forward(state.features);
    std::vector<float> probs = RLUtils::softmax(logits);
    
    if (explore) {
        return RLUtils::categorical_sample(probs);
    } else {
        auto max_it = std::max_element(probs.begin(), probs.end());
        int action = std::distance(probs.begin(), max_it);
        return Action(action);
    }
}

void TRPO::train_step(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    
    // Compute policy gradient
    std::vector<float> policy_grad(state_dim * action_dim, 0.0f);
    
    for (const auto& transition : batch) {
        float value = critic_network->forward(transition.state.features)[0];
        float next_value = transition.done ? 0.0f : 
                          critic_network->forward(transition.next_state.features)[0];
        float advantage = transition.reward + gamma * next_value - value;
        
        // Compute gradient (simplified)
    }
    
    // Compute conjugate gradient direction
    std::vector<float> search_direction = conjugate_gradient(policy_grad, batch);
    
    // Compute step size
    float step_size = std::sqrt(2.0f * delta / 
                                (std::inner_product(search_direction.begin(), search_direction.end(),
                                                  search_direction.begin(), 0.0f) + 1e-8f));
    
    // Update policy (simplified)
}

std::vector<float> TRPO::conjugate_gradient(const std::vector<float>& grad, 
                                           const std::vector<Transition>& batch) {
    // Simplified conjugate gradient implementation
    std::vector<float> x(grad.size(), 0.0f);
    std::vector<float> r = grad;
    std::vector<float> p = r;
    
    for (int i = 0; i < cg_iterations; ++i) {
        // Compute Fisher-vector product (simplified)
        std::vector<float> Fp = p;  // In practice, this would be F * p
        
        float alpha = std::inner_product(r.begin(), r.end(), r.begin(), 0.0f) /
                     (std::inner_product(p.begin(), p.end(), Fp.begin(), 0.0f) + 1e-8f);
        
        for (size_t j = 0; j < x.size(); ++j) {
            x[j] += alpha * p[j];
            r[j] -= alpha * Fp[j];
        }
        
        // Check convergence
        float r_norm = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0f));
        if (r_norm < 1e-5f) break;
        
        float beta = std::inner_product(r.begin(), r.end(), r.begin(), 0.0f) /
                    (std::inner_product(p.begin(), p.end(), p.begin(), 0.0f) + 1e-8f);
        
        for (size_t j = 0; j < p.size(); ++j) {
            p[j] = r[j] + beta * p[j];
        }
    }
    
    return x;
}

float TRPO::compute_kl_divergence(const std::vector<Transition>& batch) {
    float total_kl = 0.0f;
    
    for (const auto& transition : batch) {
        std::vector<float> logits = actor_network->forward(transition.state.features);
        std::vector<float> probs = RLUtils::softmax(logits);
        
        // KL divergence with uniform distribution (simplified)
        for (float p : probs) {
            if (p > 1e-8f) {
                total_kl += p * std::log(p * action_dim);
            }
        }
    }
    
    return total_kl / batch.size();
}

// SAC Implementation
SAC::SAC(int state_dim, int action_dim, const std::vector<int>& hidden_layers,
         float lr, float gamma, float alpha, bool auto_entropy_tuning)
    : RLAgent(state_dim, action_dim, lr, gamma), alpha(alpha), 
      auto_entropy_tuning(auto_entropy_tuning), target_entropy(-std::log(action_dim)) {
    
    actor_network = std::make_unique<DynamicNeuralNetwork>();
    actor_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        actor_network->addLayer(hidden);
    }
    actor_network->addLayer(action_dim * 2);  // Mean and log_std
    
    q1_network = std::make_unique<DynamicNeuralNetwork>();
    q1_network->addLayer(state_dim + action_dim);
    for (int hidden : hidden_layers) {
        q1_network->addLayer(hidden);
    }
    q1_network->addLayer(1);
    
    q2_network = std::make_unique<DynamicNeuralNetwork>();
    q2_network->addLayer(state_dim + action_dim);
    for (int hidden : hidden_layers) {
        q2_network->addLayer(hidden);
    }
    q2_network->addLayer(1);
    
    target_q_network = std::make_unique<DynamicNeuralNetwork>();
    target_q_network->addLayer(state_dim + action_dim);
    for (int hidden : hidden_layers) {
        target_q_network->addLayer(hidden);
    }
    target_q_network->addLayer(1);
    
    update_target_network();
}

Action SAC::select_action(const State& state, bool explore) {
    auto [action, log_prob] = sample_action_and_log_prob(state);
    return action;
}

void SAC::train_step(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    
    for (const auto& transition : batch) {
        // Sample next action and compute log_prob
        auto [next_action, next_log_prob] = sample_action_and_log_prob(transition.next_state);
        
        // Compute target Q-value
        std::vector<float> sa_next(state_dim + action_dim);
        std::copy(transition.next_state.features.begin(), transition.next_state.features.end(), sa_next.begin());
        std::copy(next_action.continuous.begin(), next_action.continuous.end(), sa_next.begin() + state_dim);
        
        float q1_next = target_q_network->forward(sa_next)[0];
        float q2_next = target_q_network->forward(sa_next)[0];
        float q_next = std::min(q1_next, q2_next) - alpha * next_log_prob;
        float target = transition.reward + gamma * (transition.done ? 0.0f : q_next);
        
        // Update Q-networks
        std::vector<float> sa(state_dim + action_dim);
        std::copy(transition.state.features.begin(), transition.state.features.end(), sa.begin());
        std::copy(transition.action.continuous.begin(), transition.action.continuous.end(), sa.begin() + state_dim);
        
        float q1 = q1_network->forward(sa)[0];
        float q2 = q2_network->forward(sa)[0];
        
        float q1_loss = (q1 - target) * (q1 - target);
        float q2_loss = (q2 - target) * (q2 - target);
        
        // Update actor
        auto [action_sample, log_prob_sample] = sample_action_and_log_prob(transition.state);
        std::vector<float> sa_sample(state_dim + action_dim);
        std::copy(transition.state.features.begin(), transition.state.features.end(), sa_sample.begin());
        std::copy(action_sample.continuous.begin(), action_sample.continuous.end(), sa_sample.begin() + state_dim);
        
        float q1_sample = q1_network->forward(sa_sample)[0];
        float actor_loss = alpha * log_prob_sample - q1_sample;
        
        // Update alpha if auto-tuning
        if (auto_entropy_tuning) {
            float alpha_loss = -alpha * (log_prob_sample + target_entropy);
            // Update alpha (simplified)
        }
    }
    
    update_target_network();
}

void SAC::update_target_network() {
    // Soft update target network
    // target = tau * current + (1 - tau) * target
}

float SAC::compute_q_value(const State& state, const Action& action, int network_id) {
    std::vector<float> sa(state_dim + action_dim);
    std::copy(state.features.begin(), state.features.end(), sa.begin());
    std::copy(action.continuous.begin(), action.continuous.end(), sa.begin() + state_dim);
    
    if (network_id == 1) {
        return q1_network->forward(sa)[0];
    } else {
        return q2_network->forward(sa)[0];
    }
}

std::pair<Action, float> SAC::sample_action_and_log_prob(const State& state) {
    std::vector<float> output = actor_network->forward(state.features);
    
    // Split into mean and log_std
    std::vector<float> mean(output.begin(), output.begin() + action_dim);
    std::vector<float> log_std(output.begin() + action_dim, output.end());
    
    // Sample action
    std::vector<float> action_continuous(action_dim);
    float log_prob = 0.0f;
    
    for (int i = 0; i < action_dim; ++i) {
        std::normal_distribution<float> dist(mean[i], std::exp(log_std[i]));
        action_continuous[i] = dist(rng);
        log_prob += -0.5f * (log_std[i] + std::log(2.0f * M_PI) + 
                            std::pow((action_continuous[i] - mean[i]) / std::exp(log_std[i]), 2));
    }
    
    return {Action(action_continuous), log_prob};
}

// Utility Functions Implementation
namespace RLUtils {
    
float compute_discounted_returns(const std::vector<float>& rewards, float gamma) {
    float discounted_return = 0.0f;
    float running_return = 0.0f;
    
    for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
        running_return = *it + gamma * running_return;
        discounted_return += running_return;
    }
    
    return discounted_return;
}

std::vector<float> compute_discounted_returns_vector(const std::vector<float>& rewards, float gamma) {
    std::vector<float> returns(rewards.size());
    float running_return = 0.0f;

    for (size_t idx = rewards.size(); idx-- > 0;) {
        running_return = rewards[idx] + gamma * running_return;
        returns[idx] = running_return;
    }

    return returns;
}

std::vector<float> compute_advantages(const std::vector<float>& rewards, 
                                     const std::vector<float>& values, float gamma, float lambda) {
    std::vector<float> advantages(rewards.size());
    float advantage = 0.0f;
    
    for (int i = rewards.size() - 1; i >= 0; --i) {
        float delta = rewards[i] + gamma * (i + 1 < values.size() ? values[i + 1] : 0.0f) - values[i];
        advantage = delta + gamma * lambda * advantage;
        advantages[i] = advantage;
    }
    
    return advantages;
}

State normalize_state(const State& state, const std::vector<float>& mean, const std::vector<float>& std) {
    State normalized(state.features.size());
    
    for (size_t i = 0; i < state.features.size(); ++i) {
        normalized.features[i] = (state.features[i] - mean[i]) / (std[i] + 1e-8f);
    }
    
    return normalized;
}

std::pair<std::vector<float>, std::vector<float>> compute_running_stats(const std::vector<State>& states) {
    if (states.empty()) {
        return {{}, {}};
    }
    
    int state_dim = states[0].features.size();
    std::vector<float> mean(state_dim, 0.0f);
    std::vector<float> std_dev(state_dim, 0.0f);
    
    // Compute mean
    for (const auto& state : states) {
        for (int i = 0; i < state_dim; ++i) {
            mean[i] += state.features[i];
        }
    }
    
    for (int i = 0; i < state_dim; ++i) {
        mean[i] /= states.size();
    }
    
    // Compute standard deviation
    for (const auto& state : states) {
        for (int i = 0; i < state_dim; ++i) {
            std_dev[i] += std::pow(state.features[i] - mean[i], 2);
        }
    }
    
    for (int i = 0; i < state_dim; ++i) {
        std_dev[i] = std::sqrt(std_dev[i] / states.size());
    }
    
    return {mean, std_dev};
}

Action epsilon_greedy_action(const std::vector<float>& q_values, float epsilon, int action_dim) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::mt19937 rng(std::random_device{}());
    
    if (dist(rng) < epsilon) {
        std::uniform_int_distribution<int> action_dist(0, action_dim - 1);
        return Action(action_dist(rng));
    } else {
        auto max_it = std::max_element(q_values.begin(), q_values.end());
        int action = std::distance(q_values.begin(), max_it);
        return Action(action);
    }
}

Action categorical_sample(const std::vector<float>& probabilities) {
    std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
    std::mt19937 rng(std::random_device{}());
    return Action(dist(rng));
}

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    
    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute softmax
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum_exp;
    }
    
    return probs;
}

float kl_divergence(const std::vector<float>& p, const std::vector<float>& q) {
    float kl = 0.0f;
    
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] > 1e-8f && q[i] > 1e-8f) {
            kl += p[i] * std::log(p[i] / q[i]);
        }
    }
    
    return kl;
}

} // namespace RLUtils

// Hierarchical RL - Options
Option::Option(int option_id, std::unique_ptr<RLAgent> policy,
               std::function<bool(const State&)> termination)
    : option_id(option_id), policy(std::move(policy)), termination_condition(termination) {}

bool Option::is_terminated(const State& state) {
    return termination_condition(state);
}

Action Option::select_action(const State& state, bool explore) {
    return policy->select_action(state, explore);
}

void Option::train_step(const std::vector<Transition>& batch) {
    policy->train_step(batch);
}

// Hierarchical Actor-Critic (HAC)
HACAgent::HACAgent(int state_dim, int action_dim, int num_options,
                   const std::vector<int>& hidden_layers, float lr, float gamma)
    : RLAgent(state_dim, action_dim, lr, gamma), current_option(0), num_options(num_options) {
    
    high_level_policy = std::make_unique<DynamicNeuralNetwork>();
    high_level_policy->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        high_level_policy->addLayer(hidden);
    }
    high_level_policy->addLayer(num_options);
}

Action HACAgent::select_action(const State& state, bool explore) {
    if (options.empty() || options[current_option]->is_terminated(state)) {
        // Select new option
        std::vector<float> option_logits = high_level_policy->forward(state.features);
        std::vector<float> option_probs = RLUtils::softmax(option_logits);
        
        if (explore) {
            current_option = RLUtils::categorical_sample(option_probs).action_id;
        } else {
            auto max_it = std::max_element(option_probs.begin(), option_probs.end());
            current_option = std::distance(option_probs.begin(), max_it);
        }
    }
    
    return options[current_option]->select_action(state, explore);
}

void HACAgent::train_step(const std::vector<Transition>& batch) {
    // Train high-level policy
    for (const auto& transition : batch) {
        std::vector<float> option_logits = high_level_policy->forward(transition.state.features);
        std::vector<float> option_probs = RLUtils::softmax(option_logits);
        
        // Update high-level policy (simplified)
    }
    
    // Train current option
    if (current_option < options.size()) {
        options[current_option]->train_step(batch);
    }
}

void HACAgent::add_option(std::unique_ptr<Option> option) {
    options.push_back(std::move(option));
}

// FeUdal Network (FuN)
FeUdalNetwork::FeUdalNetwork(int state_dim, int action_dim, int goal_dim, int horizon,
                             const std::vector<int>& hidden_layers, float lr, float gamma)
    : goal_dim(goal_dim), horizon(horizon) {
    
    manager_network = std::make_unique<DynamicNeuralNetwork>();
    manager_network->addLayer(state_dim);
    for (int hidden : hidden_layers) {
        manager_network->addLayer(hidden);
    }
    manager_network->addLayer(goal_dim);
    
    worker_network = std::make_unique<DynamicNeuralNetwork>();
    worker_network->addLayer(state_dim + goal_dim);
    for (int hidden : hidden_layers) {
        worker_network->addLayer(hidden);
    }
    worker_network->addLayer(action_dim);
}

std::vector<float> FeUdalNetwork::generate_goal(const State& state) {
    return manager_network->forward(state.features);
}

Action FeUdalNetwork::select_action(const State& state, const std::vector<float>& goal, int t) {
    std::vector<float> input(state.features.begin(), state.features.end());
    input.insert(input.end(), goal.begin(), goal.end());
    
    std::vector<float> action_values = worker_network->forward(input);
    return Action(action_values);
}

void FeUdalNetwork::train_step(const std::vector<Transition>& batch) {
    // Train manager and worker networks
    for (const auto& transition : batch) {
        // Generate goal
        std::vector<float> goal = generate_goal(transition.state);
        
        // Train manager
        float manager_reward = transition.reward;
        // Update manager network (simplified)
        
        // Train worker
        Action action = select_action(transition.state, goal, 0);
        float worker_loss = 0.0f;
        // Update worker network (simplified)
    }
}

// Offline RL - Conservative Q-Learning (CQL)
CQLAgent::CQLAgent(int state_dim, int action_dim, const std::vector<int>& hidden_layers,
                   float lr, float gamma, float alpha)
    : DQN(state_dim, action_dim, hidden_layers, lr, gamma, 1.0f, 0.995, 0.01f), alpha(alpha) {}

void CQLAgent::train_step(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    
    // Standard DQN update
    DQN::train_step(batch);
    
    // Add conservative regularization
    float cql_loss = compute_cql_loss(batch);
    
    // Update Q-network with CQL loss (simplified)
}

float CQLAgent::compute_cql_loss(const std::vector<Transition>& batch) {
    float total_cql_loss = 0.0f;
    
    for (const auto& transition : batch) {
        std::vector<float> q_values = get_q_values(transition.state);
        
        // Compute log-sum-exp of Q-values
        float max_q = *std::max_element(q_values.begin(), q_values.end());
        float sum_exp = 0.0f;
        for (float q : q_values) {
            sum_exp += std::exp(q - max_q);
        }
        float log_sum_exp = max_q + std::log(sum_exp);
        
        // CQL loss: Q(s,a) - log_sum_exp(Q(s,·))
        float q_sa = q_values[transition.action.action_id];
        float cql_loss = q_sa - log_sum_exp;
        
        total_cql_loss += alpha * cql_loss;
    }
    
    return total_cql_loss / batch.size();
}

// Batch Constrained Q-learning (BCQ)
BCQAgent::BCQAgent(int state_dim, int action_dim, const std::vector<int>& hidden_layers,
                   float lr, float gamma, float threshold)
    : DQN(state_dim, action_dim, hidden_layers, lr, gamma, 1.0f, 0.995, 0.01f), threshold(threshold) {
    
    perturbation_network = std::make_unique<DynamicNeuralNetwork>();
    perturbation_network->addLayer(state_dim + action_dim);
    for (int hidden : hidden_layers) {
        perturbation_network->addLayer(hidden);
    }
    perturbation_network->addLayer(action_dim);
}

Action BCQAgent::select_action(const State& state, bool explore) {
    // Get Q-values
    std::vector<float> q_values = get_q_values(state);
    
    // Find best action according to behavior policy (simplified - assume uniform)
    std::vector<float> behavior_probs(action_dim, 1.0f / action_dim);
    
    // Generate perturbed actions for high-probability actions
    std::vector<float> best_q_values = q_values;
    
    for (int action = 0; action < action_dim; ++action) {
        if (behavior_probs[action] > threshold) {
            Action perturbed = generate_perturbed_action(state, Action(action));
            std::vector<float> perturbed_q = get_q_values(state);
            best_q_values[action] = perturbed_q[action];
        }
    }
    
    // Select best action
    auto max_it = std::max_element(best_q_values.begin(), best_q_values.end());
    int best_action = std::distance(best_q_values.begin(), max_it);
    
    return Action(best_action);
}

void BCQAgent::train_step(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    
    // Standard DQN update
    DQN::train_step(batch);
    
    // Train perturbation network
    for (const auto& transition : batch) {
        std::vector<float> sa(transition.state.features.begin(), transition.state.features.end());
        auto action_vec = encode_action_vector(transition.action, action_dim);
        sa.insert(sa.end(), action_vec.begin(), action_vec.end());
        
        std::vector<float> perturbation = perturbation_network->forward(sa);
        
        // Compute perturbation loss (simplified)
        float perturbation_loss = 0.0f;
        for (size_t i = 0; i < perturbation.size(); ++i) {
            float error = perturbation[i];  // Simplified loss
            perturbation_loss += error * error;
        }
        
        // Update perturbation network (simplified)
    }
}

Action BCQAgent::generate_perturbed_action(const State& state, const Action& action) {
    std::vector<float> sa(state.features.begin(), state.features.end());
    auto action_vec = encode_action_vector(action, action_dim);
    sa.insert(sa.end(), action_vec.begin(), action_vec.end());
    
    std::vector<float> perturbation = perturbation_network->forward(sa);
    
    // Apply perturbation
    std::vector<float> perturbed_action = action_vec;
    for (size_t i = 0; i < perturbation.size() && i < perturbed_action.size(); ++i) {
        perturbed_action[i] += perturbation[i];
    }
    
    return Action(perturbed_action);
}

// Meta-RL - MAML
MAMLAgent::MAMLAgent(std::unique_ptr<RLAgent> base_agent, int inner_steps, float inner_lr)
    : RLAgent(base_agent->get_state_dim(), base_agent->get_action_dim(),
              base_agent->get_learning_rate(), base_agent->get_gamma()),
      base_agent(std::move(base_agent)), inner_steps(inner_steps), inner_lr(inner_lr) {}

Action MAMLAgent::select_action(const State& state, bool explore) {
    return base_agent->select_action(state, explore);
}

void MAMLAgent::train_step(const std::vector<Transition>& batch) {
    // MAML doesn't use standard train_step - use meta_update instead
}

void MAMLAgent::adapt_to_task(const std::vector<Transition>& task_batch) {
    // Inner loop adaptation
    for (int step = 0; step < inner_steps; ++step) {
        base_agent->train_step(task_batch);
    }
}

void MAMLAgent::meta_update(const std::vector<std::vector<Transition>>& tasks) {
    // Store original parameters
    // In practice, this would save actual network parameters
    
    std::vector<float> meta_gradients;
    
    for (const auto& task : tasks) {
        // Adapt to task
        adapt_to_task(task);
        
        // Compute meta-gradient
        // This is simplified - in practice would compute gradients on validation data
        
        // Restore original parameters
        // In practice, this would restore actual network parameters
    }
    
    // Update meta-parameters
    // In practice, this would update with meta-gradients
}

// Reptile Algorithm
ReptileAgent::ReptileAgent(std::unique_ptr<RLAgent> base_agent, int inner_steps, float epsilon)
    : RLAgent(base_agent->get_state_dim(), base_agent->get_action_dim(),
              base_agent->get_learning_rate(), base_agent->get_gamma()),
      base_agent(std::move(base_agent)), inner_steps(inner_steps), epsilon(epsilon) {}

Action ReptileAgent::select_action(const State& state, bool explore) {
    return base_agent->select_action(state, explore);
}

void ReptileAgent::train_step(const std::vector<Transition>& batch) {
    // Reptile doesn't use standard train_step - use meta_update instead
}

void ReptileAgent::meta_update(const std::vector<std::vector<Transition>>& tasks) {
    // Store original parameters
    std::vector<float> original_params;  // Simplified
    
    for (const auto& task : tasks) {
        // Inner loop adaptation
        for (int step = 0; step < inner_steps; ++step) {
            base_agent->train_step(task);
        }
        
        // Get adapted parameters
        std::vector<float> adapted_params;  // Simplified
        
        // Update towards adapted parameters
        for (size_t i = 0; i < original_params.size() && i < adapted_params.size(); ++i) {
            original_params[i] = original_params[i] + epsilon * (adapted_params[i] - original_params[i]);
        }
        
        // Reset to original for next task
        // In practice, this would restore actual parameters
    }
    
    // Update base agent with new parameters
    // In practice, this would set actual network parameters
}

} // namespace TinyML
