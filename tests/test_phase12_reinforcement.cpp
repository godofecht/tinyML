#include <gtest/gtest.h>
#include "ReinforcementLearning.h"
#include <random>
#include <iostream>

using namespace TinyML;

// Simple test environment
class TestEnvironment : public Environment {
private:
    int state_dim;
    int action_dim;
    std::mt19937 rng;
    
public:
    TestEnvironment(int state_dim = 4, int action_dim = 2) 
        : state_dim(state_dim), action_dim(action_dim), rng(std::random_device{}()) {}
    
    State reset() override {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        State state(state_dim);
        for (int i = 0; i < state_dim; ++i) {
            state.features[i] = dist(rng);
        }
        state.terminal = false;
        return state;
    }
    
    std::pair<State, float> step(const Action& action) override {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        State next_state(state_dim);
        for (int i = 0; i < state_dim; ++i) {
            next_state.features[i] = dist(rng);
        }
        
        // Random reward
        std::uniform_real_distribution<float> reward_dist(-1.0f, 1.0f);
        float reward = reward_dist(rng);
        
        // Random termination
        std::uniform_real_distribution<float> term_dist(0.0f, 1.0f);
        next_state.terminal = term_dist(rng) < 0.1f;
        
        return {next_state, reward};
    }
    
    int get_action_space_size() const override { return action_dim; }
    int get_state_space_size() const override { return state_dim; }
    bool is_discrete() const override { return true; }
};

class ReinforcementLearningTest : public ::testing::Test {
protected:
    void SetUp() override {
        env = std::make_unique<TestEnvironment>(4, 2);
        replay_buffer = std::make_unique<ReplayBuffer>(1000);
    }
    
    std::unique_ptr<TestEnvironment> env;
    std::unique_ptr<ReplayBuffer> replay_buffer;
};

// Test DQN
TEST_F(ReinforcementLearningTest, DQNBasicFunctionality) {
    DQN dqn(4, 2, {32, 16}, 0.001, 0.99, 1.0f, 0.995f, 0.01f);
    
    State state = env->reset();
    Action action = dqn.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    // Test training step
    std::vector<Transition> batch;
    for (int i = 0; i < 10; ++i) {
        State next_state = env->reset();
        float reward = 1.0f;
        batch.emplace_back(state, action, reward, next_state, false);
    }
    
    EXPECT_NO_THROW(dqn.train_step(batch));
    EXPECT_NO_THROW(dqn.update_target_network());
    
    // Test epsilon decay
    float initial_epsilon = dqn.get_epsilon();
    dqn.train_step(batch);
    float final_epsilon = dqn.get_epsilon();
    EXPECT_LT(final_epsilon, initial_epsilon);
}

// Test Double DQN
TEST_F(ReinforcementLearningTest, DoubleDQNBasicFunctionality) {
    DoubleDQN double_dqn(4, 2, {32, 16}, 0.001, 0.99, 1.0f, 0.995f, 0.01f);
    
    State state = env->reset();
    Action action = double_dqn.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(double_dqn.train_step(batch));
}

// Test Dueling DQN
TEST_F(ReinforcementLearningTest, DuelingDQNBasicFunctionality) {
    DuelingDQN dueling_dqn(4, 2, {32, 16}, 0.001, 0.99, 1.0f, 0.995f, 0.01f);
    
    State state = env->reset();
    std::vector<float> q_values = dueling_dqn.get_q_values(state);
    
    EXPECT_EQ(q_values.size(), 2);
    
    Action action = dueling_dqn.select_action(state, true);
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(dueling_dqn.train_step(batch));
}

// Test REINFORCE
TEST_F(ReinforcementLearningTest, REINFORCEBasicFunctionality) {
    REINFORCE reinforce(4, 2, {32, 16}, 0.001, 0.99);
    
    State state = env->reset();
    Action action = reinforce.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    reinforce.store_reward(1.0f);
    reinforce.store_reward(-0.5f);
    
    EXPECT_NO_THROW(reinforce.update_policy());
}

// Test A2C
TEST_F(ReinforcementLearningTest, A2CBasicFunctionality) {
    A2C a2c(4, 2, {32, 16}, 0.001, 0.99, 0.01f);
    
    State state = env->reset();
    Action action = a2c.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    auto [prob, value] = a2c.evaluate_state(state);
    EXPECT_GE(prob, 0.0f);
    EXPECT_LE(prob, 1.0f);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(a2c.train_step(batch));
}

// Test A3C
TEST_F(ReinforcementLearningTest, A3CBasicFunctionality) {
    A3C a3c(4, 2, 4, {32, 16}, 0.001, 0.99, 0.01f);
    
    State state = env->reset();
    Action action = a3c.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(a3c.train_step(batch));
}

// Test PPO
TEST_F(ReinforcementLearningTest, PPOBasicFunctionality) {
    PPO ppo(4, 2, {32, 16}, 0.001, 0.99, 0.2f, 0.01f, 4, 32);
    
    State state = env->reset();
    Action action = ppo.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<Transition> batch;
    for (int i = 0; i < 10; ++i) {
        State next_state = env->reset();
        batch.emplace_back(state, action, 1.0f, next_state, false);
    }
    
    EXPECT_NO_THROW(ppo.train_step(batch));
    EXPECT_NO_THROW(ppo.update_old_policy());
}

// Test TRPO
TEST_F(ReinforcementLearningTest, TRPOBasicFunctionality) {
    TRPO trpo(4, 2, {32, 16}, 0.001, 0.99, 0.01f, 10);
    
    State state = env->reset();
    Action action = trpo.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(trpo.train_step(batch));
}

// Test SAC
TEST_F(ReinforcementLearningTest, SACBasicFunctionality) {
    SAC sac(4, 2, {32, 16}, 0.001, 0.99, 0.2f, true);
    
    State state = env->reset();
    Action action = sac.select_action(state, true);
    
    EXPECT_EQ(action.action_id, -1);  // Continuous action
    EXPECT_EQ(action.continuous.size(), 2);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(sac.train_step(batch));
    EXPECT_NO_THROW(sac.update_target_network());
}

// Test World Model
TEST_F(ReinforcementLearningTest, WorldModelBasicFunctionality) {
    WorldModel world_model(4, 2, 8, {32, 16});
    
    State state = env->reset();
    Action action(1);
    
    auto [next_state, reward] = world_model.predict(state, action);
    
    EXPECT_EQ(next_state.features.size(), 4);
    EXPECT_GE(reward, -10.0f);
    EXPECT_LE(reward, 10.0f);
    
    std::vector<Transition> batch;
    for (int i = 0; i < 10; ++i) {
        State s = env->reset();
        State ns = env->reset();
        batch.emplace_back(s, action, 1.0f, ns, false);
    }
    
    EXPECT_NO_THROW(world_model.train_step(batch));
}

// Test Imagination Agent
TEST_F(ReinforcementLearningTest, ImaginationAgentBasicFunctionality) {
    auto base_agent = std::make_unique<DQN>(4, 2, std::vector<int>{32, 16}, 0.001, 0.99);
    auto world_model = std::make_unique<WorldModel>(4, 2, 8, std::vector<int>{32, 16});
    
    ImaginationAgent imagination_agent(std::move(base_agent), std::move(world_model), 5, 0.5f);
    
    State state = env->reset();
    Action action = imagination_agent.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(imagination_agent.train_step(batch));
}

// Test MADDPG Agent
TEST_F(ReinforcementLearningTest, MADDPGBasicFunctionality) {
    MADDPGAgent maddpg_agent(0, 2, 4, 2, {32, 16}, 0.001, 0.99);
    
    State state = env->reset();
    Action action = maddpg_agent.select_action(state, true);
    
    EXPECT_EQ(action.action_id, -1);  // Continuous action
    EXPECT_EQ(action.continuous.size(), 2);
    
    std::vector<std::vector<Transition>> joint_batch(2);
    State next_state = env->reset();
    joint_batch[0].emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(maddpg_agent.train_step(joint_batch));
    EXPECT_NO_THROW(maddpg_agent.update_target_network());
}

// Test QMIX Agent
TEST_F(ReinforcementLearningTest, QMIXBasicFunctionality) {
    QMIXAgent qmix_agent(2, 4, 2, {32, 16}, 0.001, 0.99);
    
    std::vector<State> states(2);
    for (int i = 0; i < 2; ++i) {
        states[i] = env->reset();
    }
    
    std::vector<Action> actions = qmix_agent.select_actions(states, true);
    EXPECT_EQ(actions.size(), 2);
    
    for (const auto& action : actions) {
        EXPECT_GE(action.action_id, 0);
        EXPECT_LT(action.action_id, 2);
    }
    
    std::vector<std::vector<Transition>> joint_batch(2);
    for (int i = 0; i < 2; ++i) {
        State next_state = env->reset();
        joint_batch[i].emplace_back(states[i], actions[i], 1.0f, next_state, false);
    }
    
    EXPECT_NO_THROW(qmix_agent.train_step(joint_batch));
    
    float qmix_value = qmix_agent.compute_qmix_value(states, actions);
    EXPECT_GE(qmix_value, -100.0f);
    EXPECT_LE(qmix_value, 100.0f);
}

// Test VDN Agent
TEST_F(ReinforcementLearningTest, VDNBasicFunctionality) {
    VDNAgent vdn_agent(2, 4, 2, {32, 16}, 0.001, 0.99);
    
    std::vector<State> states(2);
    for (int i = 0; i < 2; ++i) {
        states[i] = env->reset();
    }
    
    std::vector<Action> actions = vdn_agent.select_actions(states, true);
    EXPECT_EQ(actions.size(), 2);
    
    for (const auto& action : actions) {
        EXPECT_GE(action.action_id, 0);
        EXPECT_LT(action.action_id, 2);
    }
    
    std::vector<std::vector<Transition>> joint_batch(2);
    for (int i = 0; i < 2; ++i) {
        State next_state = env->reset();
        joint_batch[i].emplace_back(states[i], actions[i], 1.0f, next_state, false);
    }
    
    EXPECT_NO_THROW(vdn_agent.train_step(joint_batch));
    
    float vdn_value = vdn_agent.compute_vdn_value(states, actions);
    EXPECT_GE(vdn_value, -100.0f);
    EXPECT_LE(vdn_value, 100.0f);
}

// Test Option
TEST_F(ReinforcementLearningTest, OptionBasicFunctionality) {
    auto policy = std::make_unique<DQN>(4, 2, std::vector<int>{32, 16}, 0.001, 0.99);
    auto termination = [](const State& state) { return state.features[0] > 0.5f; };
    
    Option option(0, std::move(policy), termination);
    
    State state = env->reset();
    Action action = option.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    bool terminated = option.is_terminated(state);
    EXPECT_TRUE(terminated == true || terminated == false);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(option.train_step(batch));
}

// Test HAC Agent
TEST_F(ReinforcementLearningTest, HACBasicFunctionality) {
    HACAgent hac_agent(4, 2, 3, {32, 16}, 0.001, 0.99);
    
    auto policy = std::make_unique<DQN>(4, 2, std::vector<int>{16, 8}, 0.001, 0.99);
    auto termination = [](const State& state) { return state.features[0] > 0.5f; };
    auto option = std::make_unique<Option>(0, std::move(policy), termination);
    
    hac_agent.add_option(std::move(option));
    
    State state = env->reset();
    Action action = hac_agent.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(hac_agent.train_step(batch));
}

// Test FeUdal Network
TEST_F(ReinforcementLearningTest, FeUdalBasicFunctionality) {
    FeUdalNetwork feudal_network(4, 2, 8, 10, {32, 16}, 0.001, 0.99);
    
    State state = env->reset();
    std::vector<float> goal = feudal_network.generate_goal(state);
    
    EXPECT_EQ(goal.size(), 8);
    
    Action action = feudal_network.select_action(state, goal, 0);
    EXPECT_EQ(action.action_id, -1);  // Continuous action
    EXPECT_EQ(action.continuous.size(), 2);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(feudal_network.train_step(batch));
}

// Test CQL Agent
TEST_F(ReinforcementLearningTest, CQLBasicFunctionality) {
    CQLAgent cql_agent(4, 2, {32, 16}, 0.001, 0.99, 5.0f);
    
    State state = env->reset();
    Action action = cql_agent.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(cql_agent.train_step(batch));
    
    float cql_loss = cql_agent.compute_cql_loss(batch);
    EXPECT_GE(cql_loss, -100.0f);
    EXPECT_LE(cql_loss, 100.0f);
}

// Test BCQ Agent
TEST_F(ReinforcementLearningTest, BCQBasicFunctionality) {
    BCQAgent bcq_agent(4, 2, {32, 16}, 0.001, 0.99, 0.3f);
    
    State state = env->reset();
    Action action = bcq_agent.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<Transition> batch;
    State next_state = env->reset();
    batch.emplace_back(state, action, 1.0f, next_state, false);
    
    EXPECT_NO_THROW(bcq_agent.train_step(batch));
    
    Action perturbed = bcq_agent.generate_perturbed_action(state, action);
    EXPECT_EQ(perturbed.action_id, -1);  // Continuous action
    EXPECT_EQ(perturbed.continuous.size(), 2);
}

// Test MAML Agent
TEST_F(ReinforcementLearningTest, MAMLBasicFunctionality) {
    auto base_agent = std::make_unique<DQN>(4, 2, std::vector<int>{32, 16}, 0.001, 0.99);
    MAMLAgent maml_agent(std::move(base_agent), 5, 0.01f);
    
    State state = env->reset();
    Action action = maml_agent.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<Transition> task_batch;
    for (int i = 0; i < 10; ++i) {
        State s = env->reset();
        State ns = env->reset();
        task_batch.emplace_back(s, action, 1.0f, ns, false);
    }
    
    EXPECT_NO_THROW(maml_agent.adapt_to_task(task_batch));
    
    std::vector<std::vector<Transition>> tasks;
    tasks.push_back(task_batch);
    tasks.push_back(task_batch);
    
    EXPECT_NO_THROW(maml_agent.meta_update(tasks));
}

// Test Reptile Agent
TEST_F(ReinforcementLearningTest, ReptileBasicFunctionality) {
    auto base_agent = std::make_unique<DQN>(4, 2, std::vector<int>{32, 16}, 0.001, 0.99);
    ReptileAgent reptile_agent(std::move(base_agent), 5, 1.0f);
    
    State state = env->reset();
    Action action = reptile_agent.select_action(state, true);
    
    EXPECT_GE(action.action_id, 0);
    EXPECT_LT(action.action_id, 2);
    
    std::vector<std::vector<Transition>> tasks;
    std::vector<Transition> task_batch;
    for (int i = 0; i < 10; ++i) {
        State s = env->reset();
        State ns = env->reset();
        task_batch.emplace_back(s, action, 1.0f, ns, false);
    }
    tasks.push_back(task_batch);
    
    EXPECT_NO_THROW(reptile_agent.meta_update(tasks));
}

// Test Utility Functions
TEST_F(ReinforcementLearningTest, UtilityFunctions) {
    // Test discounted returns
    std::vector<float> rewards = {1.0f, 0.5f, -0.2f, 0.8f};
    float discounted_return = RLUtils::compute_discounted_returns(rewards, 0.99f);
    EXPECT_GT(discounted_return, 0.0f);
    
    // Test advantages computation
    std::vector<float> values = {0.5f, 0.3f, 0.7f, 0.2f};
    std::vector<float> advantages = RLUtils::compute_advantages(rewards, values, 0.99f, 0.95f);
    EXPECT_EQ(advantages.size(), rewards.size());
    
    // Test state normalization
    std::vector<float> mean = {0.0f, 0.5f, -0.3f, 0.2f};
    std::vector<float> std = {1.0f, 0.5f, 2.0f, 0.8f};
    State state({0.1f, 0.2f, -0.1f, 0.3f});
    State normalized_state = RLUtils::normalize_state(state, mean, std);
    EXPECT_EQ(normalized_state.features.size(), state.features.size());
    
    // Test epsilon-greedy action
    std::vector<float> q_values = {0.5f, 1.2f, -0.3f, 0.8f};
    Action eps_action = RLUtils::epsilon_greedy_action(q_values, 0.1f, 4);
    EXPECT_GE(eps_action.action_id, 0);
    EXPECT_LT(eps_action.action_id, 4);
    
    // Test categorical sampling
    std::vector<float> probs = {0.25f, 0.25f, 0.25f, 0.25f};
    Action cat_action = RLUtils::categorical_sample(probs);
    EXPECT_GE(cat_action.action_id, 0);
    EXPECT_LT(cat_action.action_id, 4);
    
    // Test softmax
    std::vector<float> logits = {1.0f, 2.0f, -1.0f, 0.5f};
    std::vector<float> softmax_probs = RLUtils::softmax(logits);
    EXPECT_EQ(softmax_probs.size(), logits.size());
    
    float prob_sum = std::accumulate(softmax_probs.begin(), softmax_probs.end(), 0.0f);
    EXPECT_NEAR(prob_sum, 1.0f, 1e-6f);
    
    // Test KL divergence
    std::vector<float> p = {0.25f, 0.25f, 0.25f, 0.25f};
    std::vector<float> q = {0.5f, 0.25f, 0.15f, 0.1f};
    float kl_div = RLUtils::kl_divergence(p, q);
    EXPECT_GE(kl_div, 0.0f);
}

// Test Replay Buffer
TEST_F(ReinforcementLearningTest, ReplayBufferBasicFunctionality) {
    ReplayBuffer buffer(100);
    
    EXPECT_EQ(buffer.size(), 0);
    
    State state = env->reset();
    Action action(1);
    State next_state = env->reset();
    
    buffer.push(Transition(state, action, 1.0f, next_state, false));
    EXPECT_EQ(buffer.size(), 1);
    
    // Fill buffer beyond capacity
    for (int i = 0; i < 150; ++i) {
        buffer.push(Transition(state, action, 1.0f, next_state, false));
    }
    
    EXPECT_EQ(buffer.size(), 100);
    
    std::vector<Transition> batch = buffer.sample(32);
    EXPECT_EQ(batch.size(), 32);
}

// Integration Test - Training Loop
TEST_F(ReinforcementLearningTest, TrainingLoopIntegration) {
    DQN dqn(4, 2, {32, 16}, 0.001, 0.99, 1.0f, 0.995f, 0.01f);
    ReplayBuffer buffer(1000);
    
    // Simulate training episodes
    for (int episode = 0; episode < 10; ++episode) {
        State state = env->reset();
        float total_reward = 0.0f;
        
        for (int step = 0; step < 20; ++step) {
            Action action = dqn.select_action(state, true);
            auto [next_state, reward] = env->step(action);
            
            buffer.push(Transition(state, action, reward, next_state, next_state.terminal));
            
            state = next_state;
            total_reward += reward;
            
            if (state.terminal) break;
        }
        
        // Train agent
        if (buffer.size() >= 32) {
            std::vector<Transition> batch = buffer.sample(32);
            dqn.train_step(batch);
        }
        
        // Update target network
        if (episode % 5 == 0) {
            dqn.update_target_network();
        }
    }
    
    // Test that epsilon has decayed
    EXPECT_LT(dqn.get_epsilon(), 1.0f);
    EXPECT_GE(dqn.get_epsilon(), 0.01f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
