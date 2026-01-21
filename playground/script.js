document.addEventListener('DOMContentLoaded', () => {
    const modelList = document.getElementById('model-list');
    const currentModelTitle = document.getElementById('current-model-title');
    const modelDescription = document.getElementById('model-description');
    const inputArea = document.getElementById('input-area');
    const runButton = document.getElementById('run-button');
    const trainButton = document.getElementById('train-button');
    const outputContent = document.getElementById('output-content');
    const statusMessage = document.getElementById('status-message');
    const chartCanvas = document.getElementById('output-chart');
    const benchmarkCanvas = document.getElementById('benchmark-chart');
    const networkCanvas = document.getElementById('network-canvas');

    const API_BASE_URL = (() => {
        const params = new URLSearchParams(window.location.search);
        return params.get('api_url') || 'http://localhost:8080';
    })();

    let currentModel = null;
    let outputChart = null;
    let benchmarkChart = null;
    let animationId = null;
    let isRunning = false;

    function resizeNetworkCanvas() {
        if (!networkCanvas) return;
        const container = networkCanvas.parentElement;
        networkCanvas.width = container.clientWidth;
        networkCanvas.height = container.clientHeight;
        drawNetwork();
    }

    window.addEventListener('resize', resizeNetworkCanvas);

    let lastWeights = null;

    function getWeightColor(weight) {
        // Sigmoid-like normalization or just linear scaling for visibility
        const magnitude = Math.abs(weight);
        const intensity = Math.min(1, Math.max(0.1, magnitude * 0.5)); // Min 0.1 alpha
        
        if (weight > 0) {
            return `rgba(0, 123, 255, ${intensity})`; // Blue
        } else {
            return `rgba(220, 53, 69, ${intensity})`; // Red
        }
    }

    function drawNetwork(weights = null) {
        if (!networkCanvas || !currentModel) return;
        
        // Update last weights if provided
        if (weights) {
            lastWeights = weights;
        } else if (weights === null && !isRunning) {
            // Use last weights if available and not explicitly cleared (optional behavior)
            weights = lastWeights;
        }

        const ctx = networkCanvas.getContext('2d');
        const width = networkCanvas.width;
        const height = networkCanvas.height;
        ctx.clearRect(0, 0, width, height);

        // Basic parameters
        const layerGap = width / 4; // Dynamic gap based on width
        const nodeRadius = Math.min(width, height) / 25; // Dynamic radius

        let layers = [];

        // Define layers based on model
        if (currentModel.id === 'perceptron') {
            const inputVal = document.getElementById('input_vector')?.value || '0,0';
            const inputSize = inputVal.split(',').filter(s => s.trim()).length || 2;
            layers = [inputSize, 3, 1];
        } else if (currentModel.id === 'bayesian') {
            const inputVal = document.getElementById('input_vector')?.value || '0,0';
            const inputSize = inputVal.split(',').filter(s => s.trim()).length || 2;
            layers = [inputSize, 5, 1];
        } else if (currentModel.id === 'generative') {
            const latentVal = document.getElementById('latent_vector')?.value || '0,0,0';
            const inputSize = latentVal.split(',').filter(s => s.trim()).length || 3;
            layers = [inputSize, 5, 10]; // Latent, Hidden, Output
        } else if (currentModel.id === 'transformer') {
            // Simplified visualization for transformer
            // Input (Sequence) -> Encoder/Decoder Blocks -> Output
            layers = [4, 4, 4, 4]; // Representing blocks
        }

        if (layers.length === 0) return;

        // Calculate positions
        const startX = (width - (layers.length - 1) * layerGap) / 2;

        // Draw connections first
        ctx.lineWidth = 1;

        let weightIndex = 0; // To track flat weight array position if needed

        for (let l = 0; l < layers.length - 1; l++) {
            const currentLayerSize = layers[l];
            const nextLayerSize = layers[l + 1];
            const currentX = startX + l * layerGap;
            const nextX = startX + (l + 1) * layerGap;
            
            // Try to get weights for this layer connection
            // For Perceptron/Bayesian: weights[l] is a vector of weights
            // But flattening logic depends on model. 
            // Simplified: if weights are provided as layer-wise 2D arrays (or 1D arrays per layer)
            let layerWeights = null;
            if (weights && weights[l]) {
                layerWeights = weights[l];
            }

            for (let i = 0; i < currentLayerSize; i++) {
                // Center the layer vertically
                const currentY = (height - (currentLayerSize - 1) * 50) / 2 + i * 50;

                for (let j = 0; j < nextLayerSize; j++) {
                    const nextY = (height - (nextLayerSize - 1) * 50) / 2 + j * 50;

                    // Determine color
                    if (layerWeights) {
                        // Assuming row-major or similar: weight for connection i -> j
                        // If layerWeights is flat: i * nextLayerSize + j
                        // If layerWeights is 2D: layerWeights[i][j] (not standard here)
                        // Let's assume flat array for the layer
                        let wVal = 0;
                        if (Array.isArray(layerWeights)) {
                             const idx = i * nextLayerSize + j;
                             if (idx < layerWeights.length) wVal = layerWeights[idx];
                        }
                        ctx.strokeStyle = getWeightColor(wVal);
                        ctx.lineWidth = 2; // Thicker for visualized weights
                    } else {
                        ctx.strokeStyle = '#999';
                        ctx.lineWidth = 1;
                    }

                    ctx.beginPath();
                    ctx.moveTo(currentX, currentY);
                    ctx.lineTo(nextX, nextY);
                    ctx.stroke();

                    // Animation: moving pulses
                    if (isRunning) {
                        const time = Date.now() / 1000;
                        const offset = (time * 2 + i * 0.2 + j * 0.3) % 1;
                        const pulseX = currentX + (nextX - currentX) * offset;
                        const pulseY = currentY + (nextY - currentY) * offset;

                        ctx.beginPath();
                        ctx.arc(pulseX, pulseY, 4, 0, Math.PI * 2);
                        ctx.fillStyle = '#007bff';
                        ctx.fill();
                    }
                }
            }
        }

        // Draw nodes
        for (let l = 0; l < layers.length; l++) {
            const layerSize = layers[l];
            const x = startX + l * layerGap;

            for (let i = 0; i < layerSize; i++) {
                const y = (height - (layerSize - 1) * 50) / 2 + i * 50;

                ctx.beginPath();
                ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
                ctx.fillStyle = '#fff';
                ctx.fill();
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            // Layer labels
            ctx.fillStyle = '#000';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            let label = "Hidden";
            if (l === 0) label = "Input";
            else if (l === layers.length - 1) label = "Output";
            
            if (currentModel.id === 'generative') {
                if (l === 0) label = "Latent";
                if (l === layers.length - 1) label = "Generated";
            }
            
            ctx.fillText(label, x, height - 20);
        }

        if (isRunning) {
            animationId = requestAnimationFrame(() => drawNetwork(weights)); // Pass weights to keep visualizing them
        }
    }

    function updateStatus(text, isError = false) {
        if (!statusMessage) {
            return;
        }
        statusMessage.textContent = text;
        statusMessage.classList.toggle('status-error', Boolean(isError));
    }

    updateStatus(`All requests go to ${API_BASE_URL}`);

    const models = [
        {
            id: 'perceptron',
            name: 'Perceptron',
            description: 'A simple single-layer neural network.',
            inputs: [
                { id: 'input_vector', name: 'Input Vector (comma-separated)', type: 'text', default: '0.5, 0.8' },
                { id: 'learning_rate', name: 'Learning Rate', type: 'number', default: 0.1 }
            ]
        },
        {
            id: 'bayesian',
            name: 'Bayesian NN',
            description: 'A neural network that uses Bayesian inference to model uncertainty.',
            inputs: [
                { id: 'input_vector', name: 'Input Vector (comma-separated)', type: 'text', default: '0.5, 0.8' },
                { id: 'dropout_rate', name: 'Dropout Rate', type: 'number', default: 0.1 },
                { id: 'mc_samples', name: 'MC Samples', type: 'number', default: 20 }
            ]
        },
        {
            id: 'generative',
            name: 'Generative Model (VAE)',
            description: 'A Variational Autoencoder that generates data from a latent vector.',
            inputs: [
                { id: 'latent_vector', name: 'Latent Vector (comma-separated)', type: 'text', default: '0.1, 0.2, 0.3' },
                { id: 'latent_dim', name: 'Latent Dimension', type: 'number', default: 3 }
            ]
        },
        {
            id: 'transformer',
            name: 'Streaming Transformer',
            description: 'A real-time transformer model for streaming data.',
            inputs: [
                { id: 'sequence', name: 'Input Sequence (comma-separated)', type: 'text', default: '0.1, 0.2, 0.3, 0.4, 0.5' }
            ]
        }
    ];

    function highlightSelected(modelId) {
        const items = modelList.querySelectorAll('li');
        items.forEach((item) => {
            item.classList.toggle('active', item.dataset.modelId === modelId);
        });
    }

    function loadModels() {
        modelList.innerHTML = '';
        models.forEach((model) => {
            const li = document.createElement('li');
            li.textContent = model.name;
            li.dataset.modelId = model.id;
            li.addEventListener('click', () => selectModel(model.id));
            modelList.appendChild(li);
        });
    }

    function selectModel(modelId) {
        currentModel = models.find((m) => m.id === modelId);
        if (!currentModel) {
            return;
        }

        highlightSelected(modelId);
        currentModelTitle.textContent = currentModel.name;
        modelDescription.textContent = currentModel.description;
        runButton.style.display = 'block';

        inputArea.innerHTML = '';
        currentModel.inputs.forEach((input) => {
            const label = document.createElement('label');
            label.htmlFor = input.id;
            label.textContent = input.name;

            const inputEl = document.createElement('input');
            inputEl.type = input.type;
            inputEl.id = input.id;
            if (input.default !== undefined) {
                inputEl.value = input.default;
            }

            inputArea.appendChild(label);
            inputArea.appendChild(inputEl);
            // Spacing handled by CSS now
        });

        // Initial draw & Resize
        resizeNetworkCanvas();
        
        // Reset charts
        renderChart(null);
        renderBenchmark(0);

        // Add listeners to update viz on input change
        const inputIds = currentModel.inputs.map(i => i.id);
        inputIds.forEach(id => {
            const el = document.getElementById(id);
            if (el && (id === 'input_vector' || id === 'latent_vector')) {
                el.addEventListener('input', () => drawNetwork(lastWeights));
            }
        });
    }

    function getVisualizationPayload(result) {
        if (!currentModel) {
            return null;
        }

        if (currentModel.id === 'perceptron') {
            const values = Array.isArray(result.output) ? result.output.map(Number) : [];
            if (!values.length) {
                return null;
            }
            return {
                label: 'Perceptron output',
                labels: values.map((_, index) => `Neuron ${index + 1}`),
                data: values
            };
        }

        if (currentModel.id === 'bayesian') {
            const mean = Number(result.mean ?? 0);
            const uncertainty = Number(result.uncertainty ?? 0);
            return {
                label: 'Bayesian stats',
                labels: ['Mean', 'Uncertainty'],
                data: [mean, uncertainty]
            };
        }

        if (currentModel.id === 'generative') {
            const values = Array.isArray(result.output) ? result.output.map(Number) : [];
            if (!values.length) {
                return null;
            }
            return {
                label: 'Generated tensor',
                labels: values.map((_, index) => `Dim ${index + 1}`),
                data: values
            };
        }

        if (currentModel.id === 'transformer') {
            const values = Array.isArray(result.output) ? result.output.map(Number) : [];
            return {
                label: 'Output Logits',
                labels: values.map((_, index) => `Token ${index + 1}`),
                data: values
            };
        }

        return null;
    }

    function renderChart(payload) {
        if (!chartCanvas || typeof Chart === 'undefined') {
            if (outputChart) {
                outputChart.destroy();
                outputChart = null;
            }
            return;
        }

        if (!payload) {
            if (outputChart) {
                outputChart.destroy();
                outputChart = null;
            }
            return;
        }

        const dataset = {
            label: payload.label,
            data: payload.data,
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
        };

        if (outputChart) {
            outputChart.data.labels = payload.labels;
            outputChart.data.datasets = [dataset];
            outputChart.update();
        } else {
            outputChart = new Chart(chartCanvas, {
                type: 'bar',
                data: {
                    labels: payload.labels,
                    datasets: [dataset]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
    }

    function renderBenchmark(myTimeUs) {
        if (!benchmarkCanvas || typeof Chart === 'undefined') {
            return;
        }

        // Reference values (microseconds)
        // These are estimates based on typical performance for similar small models
        const benchmarks = [
            { label: 'TensorFlow Lite', time: 1500, color: 'rgba(255, 99, 132, 0.5)', border: 'rgba(255, 99, 132, 1)' },
            { label: 'RTNeural', time: 350, color: 'rgba(255, 206, 86, 0.5)', border: 'rgba(255, 206, 86, 1)' },
            { label: 'ANIRA', time: 800, color: 'rgba(75, 192, 192, 0.5)', border: 'rgba(75, 192, 192, 1)' },
            { label: 'tinyML (Yours)', time: myTimeUs || 0, color: 'rgba(54, 162, 235, 0.8)', border: 'rgba(54, 162, 235, 1)' }
        ];

        const labels = benchmarks.map(b => b.label);
        const data = benchmarks.map(b => b.time);
        const backgroundColors = benchmarks.map(b => b.color);
        const borderColors = benchmarks.map(b => b.border);

        if (benchmarkChart) {
            benchmarkChart.data.datasets[0].data = data;
            benchmarkChart.update();
        } else {
            benchmarkChart = new Chart(benchmarkCanvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Inference Time (µs)',
                        data: data,
                        backgroundColor: backgroundColors,
                        borderColor: borderColors,
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y', // Horizontal bar chart
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Time (microseconds) - Lower is Better'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
    }

    runButton.addEventListener('click', async () => {
        if (!currentModel) {
            return;
        }

        const inputs = {};
        currentModel.inputs.forEach((input) => {
            const inputEl = document.getElementById(input.id);
            if (inputEl) {
                inputs[input.id] = inputEl.value;
            }
        });

        updateStatus(`Running Inference on ${currentModel.name}…`);
        outputContent.textContent = 'Running Inference…';
        
        // Start animation
        isRunning = true;
        if (animationId) cancelAnimationFrame(animationId);
        drawNetwork(lastWeights); // Keep existing weights during animation

        try {
            const response = await fetch(`${API_BASE_URL}/run/${currentModel.id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ inputs }),
                mode: 'cors'
            });

            if (!response.ok) {
                const errorBody = await response.text();
                throw new Error(`${response.status} ${response.statusText}: ${errorBody}`);
            }

            const result = await response.json();
            outputContent.textContent = JSON.stringify(result, null, 2);
            updateStatus(`Inference succeeded. Time: ${result.inference_time_us}µs`);
            const payload = getVisualizationPayload(result);
            renderChart(payload);
            renderBenchmark(result.inference_time_us);
            
            // Update weights if returned (some models might return weights on inference too)
            if (result.weights) {
                lastWeights = result.weights;
                drawNetwork(result.weights);
            }
        } catch (error) {
            outputContent.textContent = `Error: ${error.message}`;
            updateStatus(`Backend error: ${error.message}`, true);
            renderChart(null);
            renderBenchmark(null);
        } finally {
            // Stop animation after a short delay
            setTimeout(() => {
                isRunning = false;
                if (animationId) cancelAnimationFrame(animationId);
                animationId = null;
                drawNetwork(lastWeights);
            }, 1000);
        }
    });

    trainButton.addEventListener('click', async () => {
        if (!currentModel) return;

        const inputs = {};
        currentModel.inputs.forEach((input) => {
            const inputEl = document.getElementById(input.id);
            if (inputEl) inputs[input.id] = inputEl.value;
        });

        updateStatus(`Training ${currentModel.name}… (Updating weights)`);
        outputContent.textContent = 'Training…';
        
        // Start animation
        isRunning = true;
        if (animationId) cancelAnimationFrame(animationId);
        drawNetwork(lastWeights);

        try {
            const response = await fetch(`${API_BASE_URL}/train/${currentModel.id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ inputs }),
                mode: 'cors'
            });

            if (!response.ok) {
                const errorBody = await response.text();
                throw new Error(`${response.status} ${response.statusText}: ${errorBody}`);
            }

            const result = await response.json();
            outputContent.textContent = JSON.stringify(result, null, 2);
            updateStatus(`Training step completed. Weights updated.`);
            
            if (result.weights) {
                lastWeights = result.weights;
                drawNetwork(result.weights);
            }
        } catch (error) {
            outputContent.textContent = `Error: ${error.message}`;
            updateStatus(`Training failed: ${error.message}`, true);
        } finally {
            setTimeout(() => {
                isRunning = false;
                if (animationId) cancelAnimationFrame(animationId);
                animationId = null;
                drawNetwork(lastWeights);
            }, 1000);
        }
    });

    loadModels();
});
