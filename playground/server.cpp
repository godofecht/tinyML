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
