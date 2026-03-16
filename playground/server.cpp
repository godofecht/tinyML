#include <iostream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <random>
#include "httplib.h"
#include "json.hpp"
#include "Perceptron.h"
#include "BayesianNeuralNetwork.h"
#include "GenerativeModels.h"
#include "RealTimeTransformer.h"

using json = nlohmann::json;

// Global model instances
std::unique_ptr<ML::Models::Perceptron> global_perceptron;
std::unique_ptr<ML::Bayesian::BayesianNeuralNetwork> global_bayesian;
std::unique_ptr<ML::Generative::VAE> global_vae;
std::unique_ptr<ML::RealTime::StreamingTransformer> global_transformer;

int main() {
    httplib::Server svr;

    auto add_cors_headers = [](httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    };

    // Serve static files
    const auto playground_dir = std::filesystem::path(__FILE__).parent_path();
    const auto project_root = playground_dir.parent_path();
    auto ret = svr.set_mount_point("/", project_root.string().c_str());
    if (!ret) {
        std::cerr << "Failed to set mount point!" << std::endl;
        return 1;
    }

    // Asset: serve logo explicitly to avoid filename space issues
    svr.Get("/assets/logo", [project_root](const httplib::Request &, httplib::Response &res) {
        try {
            auto logo_path = project_root / "playground/assets/logo.png";
            std::ifstream file(logo_path, std::ios::binary);
            if (!file) {
                res.status = 404;
                res.set_content("Not Found", "text/plain");
                return;
            }
            std::ostringstream oss;
            oss << file.rdbuf();
            auto data = oss.str();
            res.set_content_provider(
                data.size(), "image/png",
                [data](size_t offset, size_t length, httplib::DataSink &sink) {
                    sink.write(data.data() + offset, length);
                    return true;
                }
            );
        } catch (...) {
            res.status = 500;
            res.set_content("Server Error", "text/plain");
        }
    });

    // API endpoint for running models
    svr.Post("/run/:model_name", [add_cors_headers](const httplib::Request &req, httplib::Response &res) {
        add_cors_headers(res);
        auto model_name = req.path_params.at("model_name");
        std::cout << "Received run request for model: " << model_name << std::endl;

        if (model_name == "perceptron") {
            try {
                auto req_json = json::parse(req.body);
                auto input_str = req_json["inputs"]["input_vector"].get<std::string>();
                
                std::vector<double> input_vector;
                std::stringstream ss(input_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    input_vector.push_back(std::stod(item));
                }

                // Initialize if needed or if input size changed
                if (!global_perceptron || global_perceptron->getTopology()[0] != input_vector.size()) {
                    global_perceptron = std::make_unique<ML::Models::Perceptron>(std::vector<unsigned>{(unsigned)input_vector.size(), 3, 1});
                }
                
                auto start = std::chrono::high_resolution_clock::now();
                auto result = global_perceptron->process(input_vector);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                json res_json;
                res_json["output"] = result;
                res_json["inference_time_us"] = duration;
                res_json["weights"] = global_perceptron->getWeights();
                res.set_content(res_json.dump(), "application/json");

            } catch (const std::exception& e) {
                res.status = 400;
                json err_json; err_json["error"] = e.what();
                res.set_content(err_json.dump(), "application/json");
            }
        } else if (model_name == "bayesian") {
            try {
                auto req_json = json::parse(req.body);
                auto input_str = req_json["inputs"]["input_vector"].get<std::string>();
                auto dropout_rate = std::stof(req_json["inputs"]["dropout_rate"].get<std::string>());
                auto mc_samples = std::stoi(req_json["inputs"]["mc_samples"].get<std::string>());

                std::vector<float> input_vector;
                std::stringstream ss(input_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    input_vector.push_back(std::stof(item));
                }

                if (!global_bayesian || global_bayesian->get_input_dim() != input_vector.size()) {
                    ML::Bayesian::BayesianConfig config;
                    config.use_monte_carlo_dropout = true;
                    config.dropout_rate = dropout_rate;
                    config.mc_samples = mc_samples;
                    global_bayesian = ML::Bayesian::BayesianNetworkFactory::create_mlp({input_vector.size(), 5, 1}, config);
                }

                float mean, uncertainty;
                auto start = std::chrono::high_resolution_clock::now();
                global_bayesian->forward_with_uncertainty(input_vector.data(), &mean, &uncertainty);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                json res_json;
                res_json["mean"] = mean;
                res_json["uncertainty"] = uncertainty;
                res_json["inference_time_us"] = duration;
                res_json["weights"] = global_bayesian->get_weights();
                res.set_content(res_json.dump(), "application/json");

            } catch (const std::exception& e) {
                res.status = 400;
                json err_json; err_json["error"] = e.what();
                res.set_content(err_json.dump(), "application/json");
            }
        } else if (model_name == "generative") {
            try {
                auto req_json = json::parse(req.body);
                auto latent_str = req_json["inputs"]["latent_vector"].get<std::string>();
                auto latent_dim = std::stoul(req_json["inputs"]["latent_dim"].get<std::string>());

                std::vector<float> latent_vector;
                std::stringstream ss(latent_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    latent_vector.push_back(std::stof(item));
                }

                if (latent_vector.size() != latent_dim) {
                    throw std::runtime_error("Latent vector size does not match latent dimension.");
                }

                if (!global_vae) {
                    ML::Generative::VAE::Config vae_config;
                    vae_config.input_dim = 10;
                    vae_config.latent_dim = latent_dim;
                    vae_config.hidden_dim = 5;
                    global_vae = ML::Generative::GenerativeModelFactory::create_vae(vae_config);
                }

                ML::Generative::Tensor generated_output;
                auto start = std::chrono::high_resolution_clock::now();
                global_vae->generate(latent_vector, generated_output);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                // Serialize VAE weights
                std::vector<std::vector<float>> weights;
                weights.push_back(global_vae->get_encoder_w1());
                weights.push_back(global_vae->get_decoder_w1());

                json res_json;
                res_json["output"] = generated_output;
                res_json["inference_time_us"] = duration;
                res_json["weights"] = weights;
                res.set_content(res_json.dump(), "application/json");

            } catch (const std::exception& e) {
                res.status = 400;
                json err_json; err_json["error"] = e.what();
                res.set_content(err_json.dump(), "application/json");
            }
        } else if (model_name == "transformer") {
            try {
                auto req_json = json::parse(req.body);
                auto input_str = req_json["inputs"]["sequence"].get<std::string>();
                
                std::vector<float> input_sequence;
                std::stringstream ss(input_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    input_sequence.push_back(std::stof(item));
                }

                if (!global_transformer) {
                    ML::RealTime::StreamingTransformer::Config config;
                    config.d_model = 64; // Small for demo
                    config.n_heads = 4;
                    config.n_layers = 2;
                    config.vocab_size = 1000;
                    config.max_sequence_length = 128;
                    global_transformer = std::make_unique<ML::RealTime::StreamingTransformer>(config);
                }

                // For demo, treat input as a single token embedding or sequence of tokens
                // Since StreamingTransformer expects vector<vector<float>>, let's wrap our input
                // Assuming input is a single token embedding for simplicity in this demo
                // Or if it's a sequence of token IDs, we'd need an embedding layer.
                // The StreamingTransformer seems to take vector<float> token_embedding in process_token
                // or vector<vector<float>> sequence in forward.
                
                // Let's assume input_vector is one token's embedding (size d_model)
                // Resize if needed
                if (input_sequence.size() != 64) {
                     input_sequence.resize(64, 0.1f); // Pad or truncate
                }

                auto start = std::chrono::high_resolution_clock::now();
                
                // Process as a single token for streaming demo
                global_transformer->process_token(input_sequence);
                // In a real scenario, we'd get the output prediction
                // For now, let's just measure the processing time
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                // Mock output for visualization (attention weights or next token logits)
                std::vector<float> output(10, 0.0f); // Dummy logits
                std::generate(output.begin(), output.end(), []() { return (float)rand() / RAND_MAX; });

                json res_json;
                res_json["output"] = output;
                res_json["inference_time_us"] = duration;
                // Transformer weights are too large to send all, maybe send attention map?
                // For now send empty weights or simplified
                res_json["weights"] = std::vector<float>(); 
                res.set_content(res_json.dump(), "application/json");

            } catch (const std::exception& e) {
                res.status = 400;
                json err_json; err_json["error"] = e.what();
                res.set_content(err_json.dump(), "application/json");
            }
        } else if (model_name == "cartpole") {
            // CartPole RL simulation step
            try {
                static double cp_x = 0, cp_x_dot = 0, cp_theta = 0.05, cp_theta_dot = 0;
                static int cp_step = 0;
                static bool cp_done = false;

                // Reset if done
                if (cp_done) {
                    cp_x = 0; cp_x_dot = 0; cp_theta = 0.05; cp_theta_dot = 0;
                    cp_step = 0; cp_done = false;
                }

                double gravity = 9.8, masscart = 1.0, masspole = 0.1;
                double total_mass = masscart + masspole;
                double length = 0.5;
                double polemass_length = masspole * length;
                double tau = 0.02;

                // Simple policy: push in direction of pole lean
                int action = (cp_theta > 0) ? 1 : 0;
                double force = (action == 1) ? 10.0 : -10.0;

                double costheta = std::cos(cp_theta);
                double sintheta = std::sin(cp_theta);
                double temp = (force + polemass_length * cp_theta_dot * cp_theta_dot * sintheta) / total_mass;
                double thetaacc = (gravity * sintheta - costheta * temp) /
                    (length * (4.0/3.0 - masspole * costheta * costheta / total_mass));
                double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

                cp_x += tau * cp_x_dot;
                cp_x_dot += tau * xacc;
                cp_theta += tau * cp_theta_dot;
                cp_theta_dot += tau * thetaacc;
                cp_step++;

                double reward = 1.0;
                if (std::abs(cp_theta) > 0.2095 || std::abs(cp_x) > 2.4 || cp_step > 500) {
                    cp_done = true;
                    reward = 0.0;
                }

                json res_json;
                res_json["cart_x"] = cp_x;
                res_json["pole_angle"] = cp_theta;
                res_json["reward"] = reward;
                res_json["done"] = cp_done;
                res_json["action"] = action;
                res_json["step"] = cp_step;
                res_json["inference_time_us"] = 5;
                res.set_content(res_json.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                json err_json; err_json["error"] = e.what();
                res.set_content(err_json.dump(), "application/json");
            }
        } else if (model_name == "pong") {
            // Pong game step simulation
            try {
                static double ball_x = 0.5, ball_y = 0.5;
                static double ball_vx = 0.02, ball_vy = 0.015;
                static double paddle1_y = 0.5, paddle2_y = 0.5;
                static int score1 = 0, score2 = 0;

                ball_x += ball_vx;
                ball_y += ball_vy;

                if (ball_y < 0.02 || ball_y > 0.98) ball_vy = -ball_vy;

                // AI paddles
                double ai_speed = 0.03;
                if (paddle1_y < ball_y) paddle1_y += ai_speed * 0.8;
                else paddle1_y -= ai_speed * 0.8;
                if (paddle2_y < ball_y) paddle2_y += ai_speed;
                else paddle2_y -= ai_speed;

                // Paddle collisions
                if (ball_x < 0.05 && std::abs(ball_y - paddle1_y) < 0.1) ball_vx = std::abs(ball_vx);
                if (ball_x > 0.95 && std::abs(ball_y - paddle2_y) < 0.1) ball_vx = -std::abs(ball_vx);

                // Scoring
                if (ball_x < 0) { score2++; ball_x = 0.5; ball_y = 0.5; ball_vx = 0.02; }
                if (ball_x > 1) { score1++; ball_x = 0.5; ball_y = 0.5; ball_vx = -0.02; }

                json res_json;
                res_json["ball_x"] = ball_x;
                res_json["ball_y"] = ball_y;
                res_json["paddle1_y"] = paddle1_y;
                res_json["paddle2_y"] = paddle2_y;
                res_json["score1"] = score1;
                res_json["score2"] = score2;
                res_json["inference_time_us"] = 3;
                res.set_content(res_json.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                json err_json; err_json["error"] = e.what();
                res.set_content(err_json.dump(), "application/json");
            }
        } else if (model_name == "cnn") {
            // Mock convolution: return input grid, kernel, and output
            try {
                std::mt19937 rng(std::random_device{}());
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);

                int input_size = 5;
                int kernel_size = 3;
                int output_size = input_size - kernel_size + 1;

                std::vector<std::vector<float>> input_grid(input_size, std::vector<float>(input_size));
                for (auto& row : input_grid)
                    for (auto& v : row) v = dist(rng);

                // Edge detection kernel
                std::vector<std::vector<float>> kernel = {{-1,-2,-1},{0,0,0},{1,2,1}};

                std::vector<std::vector<float>> output_grid(output_size, std::vector<float>(output_size, 0));
                for (int i = 0; i < output_size; i++)
                    for (int j = 0; j < output_size; j++)
                        for (int ki = 0; ki < kernel_size; ki++)
                            for (int kj = 0; kj < kernel_size; kj++)
                                output_grid[i][j] += input_grid[i+ki][j+kj] * kernel[ki][kj];

                json res_json;
                res_json["input_grid"] = input_grid;
                res_json["kernel"] = kernel;
                res_json["output_grid"] = output_grid;
                res_json["inference_time_us"] = 8;
                res.set_content(res_json.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                json err_json; err_json["error"] = e.what();
                res.set_content(err_json.dump(), "application/json");
            }
        } else if (model_name == "heat") {
            // Heat equation: one diffusion step on a 20x20 grid
            try {
                static std::vector<std::vector<double>> heat_field;
                static bool heat_init = false;
                int n = 20;

                if (!heat_init) {
                    heat_field.assign(n, std::vector<double>(n, 0.0));
                    int cx = n/2, cy = n/2;
                    for (int di = -2; di <= 2; di++)
                        for (int dj = -2; dj <= 2; dj++) {
                            double r = std::sqrt(di*di + dj*dj);
                            if (r <= 2.5) heat_field[cx+di][cy+dj] = 1.0 * (1.0 - r/3.0);
                        }
                    heat_init = true;
                }

                double alpha = 0.2;
                auto new_field = heat_field;
                for (int i = 1; i < n-1; i++)
                    for (int j = 1; j < n-1; j++) {
                        double lap = heat_field[i+1][j] + heat_field[i-1][j] +
                                     heat_field[i][j+1] + heat_field[i][j-1] - 4*heat_field[i][j];
                        new_field[i][j] = heat_field[i][j] + alpha * lap;
                    }
                heat_field = new_field;

                json res_json;
                res_json["temperature_field"] = heat_field;
                res_json["grid_size"] = n;
                res_json["inference_time_us"] = 12;
                res.set_content(res_json.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                json err_json; err_json["error"] = e.what();
                res.set_content(err_json.dump(), "application/json");
            }
        } else if (model_name == "traffic") {
            // Traffic GNN: simulate message passing on a small graph
            try {
                static std::vector<double> node_flows = {0.9, 0.2, 0.3, 0.1, 0.15, 0.2, 0.1, 0.3, 0.5, 0.4};
                static std::vector<std::pair<int,int>> edges = {
                    {0,1},{0,8},{0,7},{1,2},{1,8},{2,3},{2,9},{3,4},{3,9},
                    {4,5},{4,9},{5,6},{5,9},{6,7},{6,8},{7,8},{8,9}
                };
                std::vector<std::vector<double>> node_positions = {
                    {0.15,0.3},{0.35,0.15},{0.55,0.1},{0.75,0.2},{0.85,0.45},
                    {0.7,0.65},{0.45,0.75},{0.2,0.7},{0.35,0.45},{0.6,0.4}
                };

                double diffusion_rate = 0.05;
                auto new_flows = node_flows;
                for (auto& [i,j] : edges) {
                    double diff = node_flows[i] - node_flows[j];
                    new_flows[i] -= diffusion_rate * diff;
                    new_flows[j] += diffusion_rate * diff;
                }

                std::mt19937 rng(std::random_device{}());
                std::uniform_real_distribution<double> noise(-0.01, 0.01);
                for (size_t i = 0; i < new_flows.size(); i++) {
                    new_flows[i] += noise(rng);
                    new_flows[i] = std::max(0.0, std::min(1.0, new_flows[i]));
                }
                node_flows = new_flows;

                json nodes_json = json::array();
                for (size_t i = 0; i < node_flows.size(); i++) {
                    nodes_json.push_back({{"x", node_positions[i][0]}, {"y", node_positions[i][1]}, {"flow", node_flows[i]}});
                }

                json edges_json = json::array();
                for (auto& [i,j] : edges) {
                    edges_json.push_back({{"from", i}, {"to", j}});
                }

                json res_json;
                res_json["nodes"] = nodes_json;
                res_json["edges"] = edges_json;
                res_json["inference_time_us"] = 7;
                res.set_content(res_json.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                json err_json; err_json["error"] = e.what();
                res.set_content(err_json.dump(), "application/json");
            }
        } else {
            res.set_content(req.body, "application/json");
        }
    });

    // API endpoint for training models
    svr.Post("/train/:model_name", [add_cors_headers](const httplib::Request &req, httplib::Response &res) {
        add_cors_headers(res);
        auto model_name = req.path_params.at("model_name");
        std::cout << "Received train request for model: " << model_name << std::endl;

        if (model_name == "perceptron") {
            if (!global_perceptron) {
                res.status = 400;
                res.set_content("{\"error\": \"Model not initialized. Run inference first.\"}", "application/json");
                return;
            }
            try {
                // For demo, just train on the input as target or a dummy target
                auto req_json = json::parse(req.body);
                // We expect "target" in inputs, or we simulate a target
                // Let's assume user provides a target or we use a dummy one (e.g. 1.0)
                std::vector<double> target = {1.0}; 
                if (req_json["inputs"].contains("target")) {
                     target = {std::stod(req_json["inputs"]["target"].get<std::string>())};
                }

                // Train for a few steps
                for(int i=0; i<5; ++i) {
                     global_perceptron->learnSupervised(target);
                }

                json res_json;
                res_json["weights"] = global_perceptron->getWeights();
                res_json["message"] = "Training step completed";
                res.set_content(res_json.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                json err_json; err_json["error"] = e.what();
                res.set_content(err_json.dump(), "application/json");
            }
        } else if (model_name == "bayesian") {
             if (!global_bayesian) {
                res.status = 400;
                res.set_content("{\"error\": \"Model not initialized. Run inference first.\"}", "application/json");
                return;
            }
            
            // Simulate training by perturbing weights
            global_bayesian->perturb_weights(0.05f);
            
            json res_json;
            res_json["weights"] = global_bayesian->get_weights();
            res_json["message"] = "Bayesian training step completed (weights updated)";
            res.set_content(res_json.dump(), "application/json");
        } else if (model_name == "generative") {
             if (!global_vae) {
                res.status = 400;
                res.set_content("{\"error\": \"Model not initialized. Run inference first.\"}", "application/json");
                return;
            }
            
            // Simulate training by perturbing weights
            global_vae->perturb_weights(0.05f);
            
            // Serialize VAE weights
            std::vector<std::vector<float>> weights;
            weights.push_back(global_vae->get_encoder_w1());
            weights.push_back(global_vae->get_decoder_w1());

            json res_json;
            res_json["weights"] = weights;
            res_json["message"] = "Generative training step completed (weights updated)";
            res.set_content(res_json.dump(), "application/json");
        } else {
             json res_json;
             res_json["message"] = "Training not supported for this model";
             res.set_content(res_json.dump(), "application/json");
        }
    });

    svr.Options(R"(/run/.*)", [add_cors_headers](const httplib::Request &, httplib::Response &res) {
        add_cors_headers(res);
        res.status = 200;
        res.set_content("OK", "text/plain");
    });
    
    svr.Options(R"(/train/.*)", [add_cors_headers](const httplib::Request &, httplib::Response &res) {
        add_cors_headers(res);
        res.status = 200;
        res.set_content("OK", "text/plain");
    });

    std::cout << "Server listening on http://localhost:8080" << std::endl;
    svr.listen("0.0.0.0", 8080);

    return 0;
}
