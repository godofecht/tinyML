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
    const scenarioCard = document.getElementById('scenario-card');
    const scenarioTitle = document.getElementById('scenario-title');
    const scenarioCanvas = document.getElementById('scenario-canvas');

    const API_BASE_URL = (() => {
        const params = new URLSearchParams(window.location.search);
        return params.get('api_url') || 'http://localhost:8080';
    })();

    let currentModel = null;
    let outputChart = null;
    let benchmarkChart = null;
    let animationId = null;
    let isRunning = false;

    // Scenario simulation state
    let scenarioAnimationId = null;
    let scenarioInterval = null;
    let scenarioState = null;

    const SCENARIO_MODELS = ['cartpole', 'pong', 'cnn', 'heat', 'traffic'];

    function isScenarioModel(modelId) {
        return SCENARIO_MODELS.includes(modelId);
    }

    function stopScenarioAnimation() {
        if (scenarioAnimationId) {
            cancelAnimationFrame(scenarioAnimationId);
            scenarioAnimationId = null;
        }
        if (scenarioInterval) {
            clearInterval(scenarioInterval);
            scenarioInterval = null;
        }
    }

    function showScenarioCard(title) {
        const networkCard = document.querySelector('.network-card');
        if (networkCard) networkCard.style.display = 'none';
        if (scenarioCard) {
            scenarioCard.style.display = 'flex';
            scenarioTitle.textContent = title;
            // Resize canvas to fit container
            requestAnimationFrame(() => {
                const container = scenarioCanvas.parentElement;
                scenarioCanvas.width = container.clientWidth;
                scenarioCanvas.height = container.clientHeight;
            });
        }
    }

    function hideScenarioCard() {
        const networkCard = document.querySelector('.network-card');
        if (networkCard) networkCard.style.display = '';
        if (scenarioCard) scenarioCard.style.display = 'none';
        stopScenarioAnimation();
    }

    function resizeScenarioCanvas() {
        if (!scenarioCanvas || !scenarioCard || scenarioCard.style.display === 'none') return;
        const container = scenarioCanvas.parentElement;
        scenarioCanvas.width = container.clientWidth;
        scenarioCanvas.height = container.clientHeight;
    }

    // =========================================================================
    // CARTPOLE SCENARIO
    // =========================================================================
    function initCartPole() {
        scenarioState = {
            x: 0,           // cart position
            x_dot: 0,       // cart velocity
            theta: 0.05,    // pole angle (radians)
            theta_dot: 0,   // pole angular velocity
            totalReward: 0,
            step: 0,
            done: false,
            gravity: 9.8,
            massCart: 1.0,
            massPole: 0.1,
            length: 0.5,    // half-pole length
            forceMag: 10.0,
            tau: 0.02       // time step
        };
    }

    function stepCartPole() {
        const s = scenarioState;
        if (s.done) return;

        const totalMass = s.massCart + s.massPole;
        const polemassLength = s.massPole * s.length;

        // Simple policy: push right if pole leans right, left if leans left
        const force = s.theta > 0 ? s.forceMag : -s.forceMag;

        const cosTheta = Math.cos(s.theta);
        const sinTheta = Math.sin(s.theta);

        const temp = (force + polemassLength * s.theta_dot * s.theta_dot * sinTheta) / totalMass;
        const thetaAcc = (s.gravity * sinTheta - cosTheta * temp) /
            (s.length * (4.0 / 3.0 - s.massPole * cosTheta * cosTheta / totalMass));
        const xAcc = temp - polemassLength * thetaAcc * cosTheta / totalMass;

        // Euler integration
        s.x += s.tau * s.x_dot;
        s.x_dot += s.tau * xAcc;
        s.theta += s.tau * s.theta_dot;
        s.theta_dot += s.tau * thetaAcc;

        s.step++;
        s.totalReward++;

        // Terminal conditions
        if (Math.abs(s.theta) > 0.2095 || Math.abs(s.x) > 2.4 || s.step > 500) {
            s.done = true;
        }
    }

    function drawCartPole() {
        if (!scenarioCanvas || !scenarioState) return;
        const ctx = scenarioCanvas.getContext('2d');
        const W = scenarioCanvas.width;
        const H = scenarioCanvas.height;
        ctx.clearRect(0, 0, W, H);

        const s = scenarioState;
        const scale = W / 6; // map [-3, 3] meters to canvas width
        const groundY = H * 0.7;
        const cartW = scale * 0.6;
        const cartH = scale * 0.3;
        const poleLen = scale * 1.0;

        // Background
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, W, H);

        // Ground
        ctx.strokeStyle = '#4a4a6a';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, groundY + cartH / 2 + 2);
        ctx.lineTo(W, groundY + cartH / 2 + 2);
        ctx.stroke();

        // Track marks
        ctx.strokeStyle = '#3a3a5a';
        ctx.lineWidth = 1;
        for (let i = -3; i <= 3; i += 0.5) {
            const tx = W / 2 + i * scale;
            ctx.beginPath();
            ctx.moveTo(tx, groundY + cartH / 2 + 2);
            ctx.lineTo(tx, groundY + cartH / 2 + 8);
            ctx.stroke();
        }

        // Cart position on canvas
        const cartCX = W / 2 + s.x * scale;

        // Cart body
        const gradient = ctx.createLinearGradient(cartCX - cartW / 2, groundY - cartH / 2, cartCX + cartW / 2, groundY + cartH / 2);
        gradient.addColorStop(0, '#4a9eff');
        gradient.addColorStop(1, '#2a6ecf');
        ctx.fillStyle = gradient;
        ctx.fillRect(cartCX - cartW / 2, groundY - cartH / 2, cartW, cartH);
        ctx.strokeStyle = '#6ab8ff';
        ctx.lineWidth = 1;
        ctx.strokeRect(cartCX - cartW / 2, groundY - cartH / 2, cartW, cartH);

        // Wheels
        ctx.fillStyle = '#555';
        ctx.beginPath();
        ctx.arc(cartCX - cartW / 3, groundY + cartH / 2, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cartCX + cartW / 3, groundY + cartH / 2, 6, 0, Math.PI * 2);
        ctx.fill();

        // Pole
        const poleEndX = cartCX + poleLen * Math.sin(s.theta);
        const poleEndY = groundY - poleLen * Math.cos(s.theta);

        ctx.strokeStyle = s.done ? '#ff4444' : '#ffcc00';
        ctx.lineWidth = 6;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(cartCX, groundY);
        ctx.lineTo(poleEndX, poleEndY);
        ctx.stroke();

        // Pole tip
        ctx.fillStyle = s.done ? '#ff6666' : '#ffee55';
        ctx.beginPath();
        ctx.arc(poleEndX, poleEndY, 8, 0, Math.PI * 2);
        ctx.fill();

        // Pivot
        ctx.fillStyle = '#888';
        ctx.beginPath();
        ctx.arc(cartCX, groundY, 5, 0, Math.PI * 2);
        ctx.fill();

        // HUD
        ctx.fillStyle = '#ffffff';
        ctx.font = '14px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`Step: ${s.step}`, 15, 25);
        ctx.fillText(`Reward: ${s.totalReward}`, 15, 45);
        ctx.fillText(`Angle: ${(s.theta * 180 / Math.PI).toFixed(1)}deg`, 15, 65);
        ctx.fillText(`Position: ${s.x.toFixed(2)}m`, 15, 85);

        if (s.done) {
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(0, 0, W, H);
            ctx.fillStyle = '#ff4444';
            ctx.font = 'bold 28px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Episode Over', W / 2, H / 2 - 15);
            ctx.fillStyle = '#ffffff';
            ctx.font = '16px sans-serif';
            ctx.fillText(`Total Reward: ${s.totalReward} steps`, W / 2, H / 2 + 15);
            ctx.fillText('Click "Run Inference" to restart', W / 2, H / 2 + 40);
        }
    }

    function runCartPoleAnimation() {
        if (!scenarioState || scenarioState.done) {
            drawCartPole();
            return;
        }
        stepCartPole();
        drawCartPole();
        scenarioAnimationId = requestAnimationFrame(runCartPoleAnimation);
    }

    // =========================================================================
    // PONG SCENARIO
    // =========================================================================
    function initPong() {
        scenarioState = {
            ballX: 0.5,
            ballY: 0.5,
            ballVX: 0.008,
            ballVY: 0.006,
            paddle1Y: 0.5,  // left (AI)
            paddle2Y: 0.5,  // right (AI heuristic)
            score1: 0,
            score2: 0,
            paddleH: 0.15,
            paddleW: 0.02,
            ballR: 0.01,
            speed: 1,
            running: false
        };
    }

    function stepPong() {
        const s = scenarioState;
        if (!s.running) return;

        // Move ball
        s.ballX += s.ballVX * s.speed;
        s.ballY += s.ballVY * s.speed;

        // Top/bottom bounce
        if (s.ballY - s.ballR < 0) { s.ballY = s.ballR; s.ballVY = Math.abs(s.ballVY); }
        if (s.ballY + s.ballR > 1) { s.ballY = 1 - s.ballR; s.ballVY = -Math.abs(s.ballVY); }

        // AI paddles track ball with slight lag
        const aiSpeed = 0.025;
        // Left paddle (slightly slower AI)
        if (s.paddle1Y < s.ballY - 0.02) s.paddle1Y += aiSpeed * 0.8;
        else if (s.paddle1Y > s.ballY + 0.02) s.paddle1Y -= aiSpeed * 0.8;

        // Right paddle (heuristic opponent)
        if (s.paddle2Y < s.ballY - 0.01) s.paddle2Y += aiSpeed;
        else if (s.paddle2Y > s.ballY + 0.01) s.paddle2Y -= aiSpeed;

        // Clamp paddles
        s.paddle1Y = Math.max(s.paddleH / 2, Math.min(1 - s.paddleH / 2, s.paddle1Y));
        s.paddle2Y = Math.max(s.paddleH / 2, Math.min(1 - s.paddleH / 2, s.paddle2Y));

        // Left paddle collision
        if (s.ballX - s.ballR < s.paddleW + 0.02) {
            if (s.ballY > s.paddle1Y - s.paddleH / 2 && s.ballY < s.paddle1Y + s.paddleH / 2) {
                s.ballX = s.paddleW + 0.02 + s.ballR;
                s.ballVX = Math.abs(s.ballVX) * 1.02;
                // Add spin based on where ball hits paddle
                const relHit = (s.ballY - s.paddle1Y) / (s.paddleH / 2);
                s.ballVY += relHit * 0.003;
            }
        }

        // Right paddle collision
        if (s.ballX + s.ballR > 1 - s.paddleW - 0.02) {
            if (s.ballY > s.paddle2Y - s.paddleH / 2 && s.ballY < s.paddle2Y + s.paddleH / 2) {
                s.ballX = 1 - s.paddleW - 0.02 - s.ballR;
                s.ballVX = -Math.abs(s.ballVX) * 1.02;
                const relHit = (s.ballY - s.paddle2Y) / (s.paddleH / 2);
                s.ballVY += relHit * 0.003;
            }
        }

        // Speed cap
        const maxV = 0.02;
        s.ballVX = Math.max(-maxV, Math.min(maxV, s.ballVX));
        s.ballVY = Math.max(-maxV, Math.min(maxV, s.ballVY));

        // Scoring
        if (s.ballX < 0) {
            s.score2++;
            resetPongBall(s, 1);
        }
        if (s.ballX > 1) {
            s.score1++;
            resetPongBall(s, -1);
        }
    }

    function resetPongBall(s, direction) {
        s.ballX = 0.5;
        s.ballY = 0.5;
        s.ballVX = 0.008 * direction;
        s.ballVY = (Math.random() - 0.5) * 0.01;
    }

    function drawPong() {
        if (!scenarioCanvas || !scenarioState) return;
        const ctx = scenarioCanvas.getContext('2d');
        const W = scenarioCanvas.width;
        const H = scenarioCanvas.height;
        const s = scenarioState;

        // Background
        ctx.fillStyle = '#0a0a1e';
        ctx.fillRect(0, 0, W, H);

        // Center line
        ctx.setLineDash([8, 8]);
        ctx.strokeStyle = '#333355';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(W / 2, 0);
        ctx.lineTo(W / 2, H);
        ctx.stroke();
        ctx.setLineDash([]);

        // Center circle
        ctx.strokeStyle = '#333355';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(W / 2, H / 2, 40, 0, Math.PI * 2);
        ctx.stroke();

        // Paddles
        const pw = s.paddleW * W;
        const ph = s.paddleH * H;

        // Left paddle (RL Agent - blue)
        const p1Gradient = ctx.createLinearGradient(0.02 * W, 0, 0.02 * W + pw, 0);
        p1Gradient.addColorStop(0, '#4a9eff');
        p1Gradient.addColorStop(1, '#2a6ecf');
        ctx.fillStyle = p1Gradient;
        ctx.fillRect(0.02 * W, s.paddle1Y * H - ph / 2, pw, ph);

        // Right paddle (Heuristic - orange)
        const p2Gradient = ctx.createLinearGradient(W - 0.02 * W - pw, 0, W - 0.02 * W, 0);
        p2Gradient.addColorStop(0, '#ff8c42');
        p2Gradient.addColorStop(1, '#cf6c22');
        ctx.fillStyle = p2Gradient;
        ctx.fillRect(W - 0.02 * W - pw, s.paddle2Y * H - ph / 2, pw, ph);

        // Ball with glow
        const bx = s.ballX * W;
        const by = s.ballY * H;
        const br = s.ballR * W;

        ctx.shadowColor = '#ffffff';
        ctx.shadowBlur = 15;
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(bx, by, br, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Ball trail effect
        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.beginPath();
        ctx.arc(bx - s.ballVX * W * 3, by - s.ballVY * H * 3, br * 0.7, 0, Math.PI * 2);
        ctx.fill();

        // Scores
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 48px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(s.score1, W / 2 - 60, 55);
        ctx.fillText(s.score2, W / 2 + 60, 55);

        // Labels
        ctx.font = '12px sans-serif';
        ctx.fillStyle = '#4a9eff';
        ctx.fillText('RL Agent', W * 0.1, H - 15);
        ctx.fillStyle = '#ff8c42';
        ctx.fillText('Heuristic', W * 0.9, H - 15);

        if (!s.running) {
            ctx.fillStyle = 'rgba(0,0,0,0.4)';
            ctx.fillRect(0, 0, W, H);
            ctx.fillStyle = '#ffffff';
            ctx.font = '20px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Click "Run Inference" to start', W / 2, H / 2);
        }
    }

    function runPongAnimation() {
        if (!scenarioState) return;
        stepPong();
        drawPong();
        if (scenarioState.running) {
            scenarioAnimationId = requestAnimationFrame(runPongAnimation);
        }
    }

    // =========================================================================
    // CNN SCENARIO
    // =========================================================================
    function initCNN() {
        const inputSize = 7;
        const kernelSize = 3;
        const outputSize = inputSize - kernelSize + 1;

        // Random input grid
        const inputGrid = [];
        for (let i = 0; i < inputSize; i++) {
            inputGrid[i] = [];
            for (let j = 0; j < inputSize; j++) {
                inputGrid[i][j] = Math.random();
            }
        }

        // Edge detection kernel (Sobel-like horizontal)
        const kernel = [
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ];

        // Compute convolution
        const outputGrid = [];
        for (let i = 0; i < outputSize; i++) {
            outputGrid[i] = [];
            for (let j = 0; j < outputSize; j++) {
                let sum = 0;
                for (let ki = 0; ki < kernelSize; ki++) {
                    for (let kj = 0; kj < kernelSize; kj++) {
                        sum += inputGrid[i + ki][j + kj] * kernel[ki][kj];
                    }
                }
                outputGrid[i][j] = sum;
            }
        }

        scenarioState = {
            inputGrid,
            kernel,
            outputGrid,
            inputSize,
            kernelSize,
            outputSize,
            highlightI: -1,
            highlightJ: -1,
            animStep: 0,
            animating: false,
            kernelNames: ['Horizontal Edge', 'Vertical Edge', 'Sharpen', 'Blur'],
            kernels: [
                [[-1,-2,-1],[0,0,0],[1,2,1]],
                [[-1,0,1],[-2,0,2],[-1,0,1]],
                [[0,-1,0],[-1,5,-1],[0,-1,0]],
                [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
            ],
            currentKernel: 0
        };
    }

    function recomputeCNN() {
        const s = scenarioState;
        const outputGrid = [];
        for (let i = 0; i < s.outputSize; i++) {
            outputGrid[i] = [];
            for (let j = 0; j < s.outputSize; j++) {
                let sum = 0;
                for (let ki = 0; ki < s.kernelSize; ki++) {
                    for (let kj = 0; kj < s.kernelSize; kj++) {
                        sum += s.inputGrid[i + ki][j + kj] * s.kernel[ki][kj];
                    }
                }
                outputGrid[i][j] = sum;
            }
        }
        s.outputGrid = outputGrid;
    }

    function drawCNN() {
        if (!scenarioCanvas || !scenarioState) return;
        const ctx = scenarioCanvas.getContext('2d');
        const W = scenarioCanvas.width;
        const H = scenarioCanvas.height;
        const s = scenarioState;

        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, W, H);

        const padding = 30;
        const totalGrids = 3;
        const availW = W - padding * 4;
        const maxCellSize = Math.min(
            availW / (s.inputSize + s.kernelSize + s.outputSize + 4),
            (H - 120) / Math.max(s.inputSize, s.kernelSize, s.outputSize)
        );
        const cellSize = Math.max(12, Math.min(40, maxCellSize));

        // Normalize output for coloring
        let minOut = Infinity, maxOut = -Infinity;
        for (let i = 0; i < s.outputSize; i++) {
            for (let j = 0; j < s.outputSize; j++) {
                minOut = Math.min(minOut, s.outputGrid[i][j]);
                maxOut = Math.max(maxOut, s.outputGrid[i][j]);
            }
        }
        const outRange = maxOut - minOut || 1;

        function getHeatColor(val, min, max) {
            const t = (val - min) / (max - min || 1);
            const r = Math.floor(255 * Math.min(1, t * 2));
            const g = Math.floor(255 * Math.min(1, (1 - Math.abs(t - 0.5) * 2)));
            const b = Math.floor(255 * Math.min(1, (1 - t) * 2));
            return `rgb(${r},${g},${b})`;
        }

        // Layout: input | * | kernel | = | output
        const inputW = s.inputSize * cellSize;
        const kernelW = s.kernelSize * cellSize;
        const outputW = s.outputSize * cellSize;
        const totalW = inputW + kernelW + outputW + 120;
        const startX = (W - totalW) / 2;

        const baseY = 70;

        // Draw grid helper
        function drawGrid(grid, rows, cols, ox, oy, colorFn) {
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    const x = ox + j * cellSize;
                    const y = oy + i * cellSize;
                    ctx.fillStyle = colorFn(grid[i][j]);
                    ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

                    // Value text if cells are big enough
                    if (cellSize >= 28) {
                        ctx.fillStyle = '#fff';
                        ctx.font = '10px monospace';
                        ctx.textAlign = 'center';
                        ctx.fillText(grid[i][j].toFixed(1), x + cellSize / 2, y + cellSize / 2 + 4);
                    }
                }
            }
        }

        // Highlight the current convolution window
        const hi = s.highlightI;
        const hj = s.highlightJ;

        // Input grid
        const inputX = startX;
        const inputY = baseY;
        ctx.fillStyle = '#cccccc';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Input (' + s.inputSize + 'x' + s.inputSize + ')', inputX + inputW / 2, inputY - 10);
        drawGrid(s.inputGrid, s.inputSize, s.inputSize, inputX, inputY, (v) => {
            const t = v;
            return `rgb(${Math.floor(50 + 180 * t)}, ${Math.floor(50 + 180 * t)}, ${Math.floor(80 + 160 * t)})`;
        });

        // Highlight convolution window on input
        if (hi >= 0 && hj >= 0) {
            ctx.strokeStyle = '#ffcc00';
            ctx.lineWidth = 3;
            ctx.strokeRect(inputX + hj * cellSize - 1, inputY + hi * cellSize - 1, s.kernelSize * cellSize + 1, s.kernelSize * cellSize + 1);
        }

        // Operator symbol *
        const opX = inputX + inputW + 25;
        const opY = baseY + (s.inputSize * cellSize) / 2;
        ctx.fillStyle = '#ffcc00';
        ctx.font = 'bold 24px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('*', opX, opY + 8);

        // Kernel
        const kernelX = opX + 30;
        const kernelY = baseY + (s.inputSize * cellSize - s.kernelSize * cellSize) / 2;
        ctx.fillStyle = '#cccccc';
        ctx.font = '14px sans-serif';
        ctx.fillText(s.kernelNames[s.currentKernel] + ' Kernel', kernelX + kernelW / 2, kernelY - 10);
        drawGrid(s.kernel, s.kernelSize, s.kernelSize, kernelX, kernelY, (v) => {
            if (v > 0) return `rgba(100, 200, 100, ${Math.min(1, Math.abs(v) * 0.3 + 0.2)})`;
            if (v < 0) return `rgba(200, 100, 100, ${Math.min(1, Math.abs(v) * 0.3 + 0.2)})`;
            return 'rgba(100, 100, 100, 0.3)';
        });

        // Operator symbol =
        const eqX = kernelX + kernelW + 25;
        ctx.fillStyle = '#ffcc00';
        ctx.font = 'bold 24px sans-serif';
        ctx.fillText('=', eqX, opY + 8);

        // Output grid
        const outX = eqX + 30;
        const outY = baseY + (s.inputSize * cellSize - s.outputSize * cellSize) / 2;
        ctx.fillStyle = '#cccccc';
        ctx.font = '14px sans-serif';
        ctx.fillText('Feature Map (' + s.outputSize + 'x' + s.outputSize + ')', outX + outputW / 2, outY - 10);
        drawGrid(s.outputGrid, s.outputSize, s.outputSize, outX, outY, (v) => getHeatColor(v, minOut, maxOut));

        // Highlight output cell
        if (hi >= 0 && hj >= 0) {
            ctx.strokeStyle = '#ffcc00';
            ctx.lineWidth = 3;
            ctx.strokeRect(outX + hj * cellSize - 1, outY + hi * cellSize - 1, cellSize + 1, cellSize + 1);
        }

        // Instructions
        ctx.fillStyle = '#8888aa';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Click "Run Inference" to animate convolution with next kernel', W / 2, H - 15);
    }

    function animateCNNConvolution() {
        const s = scenarioState;
        if (!s.animating) return;

        const totalSteps = s.outputSize * s.outputSize;
        if (s.animStep >= totalSteps) {
            s.animating = false;
            s.highlightI = -1;
            s.highlightJ = -1;
            drawCNN();
            return;
        }

        s.highlightI = Math.floor(s.animStep / s.outputSize);
        s.highlightJ = s.animStep % s.outputSize;
        s.animStep++;
        drawCNN();

        scenarioInterval = setTimeout(() => animateCNNConvolution(), 200);
    }

    // =========================================================================
    // HEAT EQUATION (PINN) SCENARIO
    // =========================================================================
    function initHeat() {
        const gridSize = 30;
        const field = [];
        for (let i = 0; i < gridSize; i++) {
            field[i] = [];
            for (let j = 0; j < gridSize; j++) {
                field[i][j] = 0;
            }
        }
        // Hot spot in center
        const cx = Math.floor(gridSize / 2);
        const cy = Math.floor(gridSize / 2);
        for (let di = -2; di <= 2; di++) {
            for (let dj = -2; dj <= 2; dj++) {
                const r = Math.sqrt(di * di + dj * dj);
                if (r <= 2.5) {
                    field[cx + di][cy + dj] = 1.0 * (1 - r / 3);
                }
            }
        }

        scenarioState = {
            field,
            gridSize,
            step: 0,
            alpha: 0.2,  // diffusion coefficient
            running: false,
            maxTemp: 1.0
        };
    }

    function stepHeat() {
        const s = scenarioState;
        const n = s.gridSize;
        const newField = [];
        for (let i = 0; i < n; i++) {
            newField[i] = [];
            for (let j = 0; j < n; j++) {
                if (i === 0 || j === 0 || i === n - 1 || j === n - 1) {
                    newField[i][j] = 0; // boundary condition
                } else {
                    // Discrete Laplacian
                    const laplacian = s.field[i + 1][j] + s.field[i - 1][j] +
                        s.field[i][j + 1] + s.field[i][j - 1] - 4 * s.field[i][j];
                    newField[i][j] = s.field[i][j] + s.alpha * laplacian;
                }
            }
        }
        s.field = newField;
        s.step++;
    }

    function drawHeat() {
        if (!scenarioCanvas || !scenarioState) return;
        const ctx = scenarioCanvas.getContext('2d');
        const W = scenarioCanvas.width;
        const H = scenarioCanvas.height;
        const s = scenarioState;

        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, W, H);

        const n = s.gridSize;
        const padding = 50;
        const availSize = Math.min(W - padding * 2, H - 100);
        const cellSize = Math.floor(availSize / n);
        const gridW = cellSize * n;
        const ox = (W - gridW) / 2;
        const oy = 50;

        // Find max for normalization
        let maxVal = 0.001;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                maxVal = Math.max(maxVal, Math.abs(s.field[i][j]));
            }
        }

        // Draw heatmap
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const t = Math.max(0, Math.min(1, s.field[i][j] / maxVal));
                // Blue (cold) -> Yellow -> Red (hot)
                let r, g, b;
                if (t < 0.25) {
                    const lt = t / 0.25;
                    r = 10; g = Math.floor(20 + 60 * lt); b = Math.floor(80 + 175 * lt);
                } else if (t < 0.5) {
                    const lt = (t - 0.25) / 0.25;
                    r = Math.floor(20 * lt); g = Math.floor(80 + 175 * lt); b = 255;
                } else if (t < 0.75) {
                    const lt = (t - 0.5) / 0.25;
                    r = Math.floor(20 + 235 * lt); g = 255; b = Math.floor(255 * (1 - lt));
                } else {
                    const lt = (t - 0.75) / 0.25;
                    r = 255; g = Math.floor(255 * (1 - lt)); b = 0;
                }
                ctx.fillStyle = `rgb(${r},${g},${b})`;
                ctx.fillRect(ox + j * cellSize, oy + i * cellSize, cellSize, cellSize);
            }
        }

        // Grid border
        ctx.strokeStyle = '#444466';
        ctx.lineWidth = 1;
        ctx.strokeRect(ox, oy, gridW, gridW);

        // Title and HUD
        ctx.fillStyle = '#ffffff';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Heat Equation (PINN) - 2D Diffusion', W / 2, 25);

        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`Step: ${s.step}`, ox, oy + gridW + 25);
        ctx.fillText(`Max Temp: ${maxVal.toFixed(4)}`, ox, oy + gridW + 42);
        ctx.fillText(`Alpha: ${s.alpha}`, ox + 200, oy + gridW + 25);

        // Color bar legend
        const barX = ox + gridW + 15;
        const barY = oy;
        const barW = 20;
        const barH = gridW;
        if (barX + barW < W) {
            for (let py = 0; py < barH; py++) {
                const t = 1 - py / barH;
                let r, g, b;
                if (t < 0.25) {
                    const lt = t / 0.25;
                    r = 10; g = Math.floor(20 + 60 * lt); b = Math.floor(80 + 175 * lt);
                } else if (t < 0.5) {
                    const lt = (t - 0.25) / 0.25;
                    r = Math.floor(20 * lt); g = Math.floor(80 + 175 * lt); b = 255;
                } else if (t < 0.75) {
                    const lt = (t - 0.5) / 0.25;
                    r = Math.floor(20 + 235 * lt); g = 255; b = Math.floor(255 * (1 - lt));
                } else {
                    const lt = (t - 0.75) / 0.25;
                    r = 255; g = Math.floor(255 * (1 - lt)); b = 0;
                }
                ctx.fillStyle = `rgb(${r},${g},${b})`;
                ctx.fillRect(barX, barY + py, barW, 1);
            }
            ctx.strokeStyle = '#666';
            ctx.strokeRect(barX, barY, barW, barH);
            ctx.fillStyle = '#aaa';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText('Hot', barX + barW + 4, barY + 10);
            ctx.fillText('Cold', barX + barW + 4, barY + barH);
        }

        if (!s.running) {
            ctx.fillStyle = '#8888aa';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Click "Run Inference" to start diffusion', W / 2, H - 10);
        }
    }

    function runHeatAnimation() {
        if (!scenarioState || !scenarioState.running) return;
        for (let i = 0; i < 3; i++) stepHeat(); // multiple sub-steps per frame
        drawHeat();
        scenarioAnimationId = requestAnimationFrame(runHeatAnimation);
    }

    // =========================================================================
    // TRAFFIC (GNN) SCENARIO
    // =========================================================================
    function initTraffic() {
        // Create a small road network graph
        const nodes = [
            { x: 0.15, y: 0.3, flow: 0.9, label: 'A' },
            { x: 0.35, y: 0.15, flow: 0.2, label: 'B' },
            { x: 0.55, y: 0.1, flow: 0.3, label: 'C' },
            { x: 0.75, y: 0.2, flow: 0.1, label: 'D' },
            { x: 0.85, y: 0.45, flow: 0.15, label: 'E' },
            { x: 0.7, y: 0.65, flow: 0.2, label: 'F' },
            { x: 0.45, y: 0.75, flow: 0.1, label: 'G' },
            { x: 0.2, y: 0.7, flow: 0.3, label: 'H' },
            { x: 0.35, y: 0.45, flow: 0.5, label: 'I' },
            { x: 0.6, y: 0.4, flow: 0.4, label: 'J' }
        ];

        // Edges (road connections)
        const edges = [
            [0, 1], [0, 8], [0, 7],
            [1, 2], [1, 8],
            [2, 3], [2, 9],
            [3, 4], [3, 9],
            [4, 5], [4, 9],
            [5, 6], [5, 9],
            [6, 7], [6, 8],
            [7, 8],
            [8, 9]
        ];

        // Particles for edge animation
        const particles = [];
        edges.forEach((e, idx) => {
            particles.push({ edge: idx, t: Math.random(), speed: 0.005 + Math.random() * 0.01 });
        });

        scenarioState = {
            nodes,
            edges,
            particles,
            step: 0,
            running: false,
            diffusionRate: 0.05
        };
    }

    function stepTraffic() {
        const s = scenarioState;
        const n = s.nodes.length;

        // GNN-style message passing: each node aggregates neighbor flows
        const newFlows = s.nodes.map(node => node.flow);

        for (const [i, j] of s.edges) {
            const diff = s.nodes[i].flow - s.nodes[j].flow;
            newFlows[i] -= s.diffusionRate * diff;
            newFlows[j] += s.diffusionRate * diff;
        }

        // Add small random perturbation (simulating new traffic entering)
        for (let i = 0; i < n; i++) {
            newFlows[i] += (Math.random() - 0.5) * 0.02;
            newFlows[i] = Math.max(0, Math.min(1, newFlows[i]));
            s.nodes[i].flow = newFlows[i];
        }

        // Move particles
        for (const p of s.particles) {
            p.t += p.speed;
            if (p.t > 1) p.t -= 1;
        }

        s.step++;
    }

    function drawTraffic() {
        if (!scenarioCanvas || !scenarioState) return;
        const ctx = scenarioCanvas.getContext('2d');
        const W = scenarioCanvas.width;
        const H = scenarioCanvas.height;
        const s = scenarioState;

        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, W, H);

        const pad = 40;
        const gW = W - pad * 2;
        const gH = H - 100;

        function nx(x) { return pad + x * gW; }
        function ny(y) { return 50 + y * gH; }

        // Draw edges
        for (let ei = 0; ei < s.edges.length; ei++) {
            const [i, j] = s.edges[ei];
            const avgFlow = (s.nodes[i].flow + s.nodes[j].flow) / 2;
            const lineWidth = 1 + avgFlow * 4;

            // Color based on congestion
            const r = Math.floor(255 * avgFlow);
            const g = Math.floor(255 * (1 - avgFlow));
            ctx.strokeStyle = `rgba(${r}, ${g}, 80, 0.6)`;
            ctx.lineWidth = lineWidth;
            ctx.beginPath();
            ctx.moveTo(nx(s.nodes[i].x), ny(s.nodes[i].y));
            ctx.lineTo(nx(s.nodes[j].x), ny(s.nodes[j].y));
            ctx.stroke();
        }

        // Draw particles along edges
        for (const p of s.particles) {
            const [i, j] = s.edges[p.edge];
            const x1 = nx(s.nodes[i].x);
            const y1 = ny(s.nodes[i].y);
            const x2 = nx(s.nodes[j].x);
            const y2 = ny(s.nodes[j].y);
            const px = x1 + (x2 - x1) * p.t;
            const py = y1 + (y2 - y1) * p.t;

            const avgFlow = (s.nodes[i].flow + s.nodes[j].flow) / 2;
            ctx.fillStyle = `rgba(255, 255, 100, ${0.3 + avgFlow * 0.7})`;
            ctx.beginPath();
            ctx.arc(px, py, 2 + avgFlow * 2, 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw nodes
        for (let i = 0; i < s.nodes.length; i++) {
            const node = s.nodes[i];
            const x = nx(node.x);
            const y = ny(node.y);
            const radius = 12 + node.flow * 18;

            // Glow
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 1.5);
            const flowR = Math.floor(255 * node.flow);
            const flowG = Math.floor(255 * (1 - node.flow));
            gradient.addColorStop(0, `rgba(${flowR}, ${flowG}, 80, 0.8)`);
            gradient.addColorStop(1, `rgba(${flowR}, ${flowG}, 80, 0)`);
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, radius * 1.5, 0, Math.PI * 2);
            ctx.fill();

            // Node circle
            ctx.fillStyle = `rgb(${flowR}, ${flowG}, 80)`;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Label
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(node.label, x, y + 4);

            // Flow value
            ctx.fillStyle = '#aaa';
            ctx.font = '10px monospace';
            ctx.fillText(node.flow.toFixed(2), x, y + radius + 14);
        }

        // Title
        ctx.fillStyle = '#ffffff';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Traffic Flow (GNN Message Passing)', W / 2, 25);

        // HUD
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`Step: ${s.step}`, 15, H - 30);
        ctx.fillStyle = '#88aa88';
        ctx.fillText('Green = low traffic', 15, H - 12);
        ctx.fillStyle = '#aa8888';
        ctx.fillText('Red = high traffic', 180, H - 12);

        if (!s.running) {
            ctx.fillStyle = '#8888aa';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Click "Run Inference" to start simulation', W / 2, H - 50);
        }
    }

    function runTrafficAnimation() {
        if (!scenarioState || !scenarioState.running) return;
        stepTraffic();
        drawTraffic();
        scenarioAnimationId = requestAnimationFrame(runTrafficAnimation);
    }

    // =========================================================================
    // ORIGINAL CODE (preserved)
    // =========================================================================

    function resizeNetworkCanvas() {
        if (!networkCanvas) return;
        const container = networkCanvas.parentElement;
        networkCanvas.width = container.clientWidth;
        networkCanvas.height = container.clientHeight;
        drawNetwork();
    }

    window.addEventListener('resize', () => {
        resizeNetworkCanvas();
        resizeScenarioCanvas();
        // Redraw current scenario if active
        if (currentModel && isScenarioModel(currentModel.id) && scenarioState) {
            resizeScenarioCanvas();
            requestAnimationFrame(() => {
                if (currentModel.id === 'cartpole') drawCartPole();
                else if (currentModel.id === 'pong') drawPong();
                else if (currentModel.id === 'cnn') drawCNN();
                else if (currentModel.id === 'heat') drawHeat();
                else if (currentModel.id === 'traffic') drawTraffic();
            });
        }
    });

    let lastWeights = null;

    function getWeightColor(weight) {
        const magnitude = Math.abs(weight);
        const intensity = Math.min(1, Math.max(0.1, magnitude * 0.5));

        if (weight > 0) {
            return `rgba(0, 123, 255, ${intensity})`;
        } else {
            return `rgba(220, 53, 69, ${intensity})`;
        }
    }

    function drawNetwork(weights = null) {
        if (!networkCanvas || !currentModel) return;
        // Skip for scenario models
        if (isScenarioModel(currentModel.id)) return;

        if (weights) {
            lastWeights = weights;
        } else if (weights === null && !isRunning) {
            weights = lastWeights;
        }

        const ctx = networkCanvas.getContext('2d');
        const width = networkCanvas.width;
        const height = networkCanvas.height;
        ctx.clearRect(0, 0, width, height);

        const layerGap = width / 4;
        const nodeRadius = Math.min(width, height) / 25;

        let layers = [];

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
            layers = [inputSize, 5, 10];
        } else if (currentModel.id === 'transformer') {
            layers = [4, 4, 4, 4];
        }

        if (layers.length === 0) return;

        const startX = (width - (layers.length - 1) * layerGap) / 2;

        ctx.lineWidth = 1;

        let weightIndex = 0;

        for (let l = 0; l < layers.length - 1; l++) {
            const currentLayerSize = layers[l];
            const nextLayerSize = layers[l + 1];
            const currentX = startX + l * layerGap;
            const nextX = startX + (l + 1) * layerGap;

            let layerWeights = null;
            if (weights && weights[l]) {
                layerWeights = weights[l];
            }

            for (let i = 0; i < currentLayerSize; i++) {
                const currentY = (height - (currentLayerSize - 1) * 50) / 2 + i * 50;

                for (let j = 0; j < nextLayerSize; j++) {
                    const nextY = (height - (nextLayerSize - 1) * 50) / 2 + j * 50;

                    if (layerWeights) {
                        let wVal = 0;
                        if (Array.isArray(layerWeights)) {
                             const idx = i * nextLayerSize + j;
                             if (idx < layerWeights.length) wVal = layerWeights[idx];
                        }
                        ctx.strokeStyle = getWeightColor(wVal);
                        ctx.lineWidth = 2;
                    } else {
                        ctx.strokeStyle = '#999';
                        ctx.lineWidth = 1;
                    }

                    ctx.beginPath();
                    ctx.moveTo(currentX, currentY);
                    ctx.lineTo(nextX, nextY);
                    ctx.stroke();

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
            animationId = requestAnimationFrame(() => drawNetwork(weights));
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
        },
        // ---- NEW SCENARIO MODELS ----
        {
            id: 'cartpole',
            name: 'CartPole (RL)',
            description: 'Reinforcement Learning agent balances a pole on a cart. The agent applies force left/right to keep the pole upright. Watch the physics simulation in real time.',
            scenario: true,
            inputs: [
                { id: 'cp_force', name: 'Force Magnitude', type: 'number', default: 10.0 },
                { id: 'cp_gravity', name: 'Gravity', type: 'number', default: 9.8 }
            ]
        },
        {
            id: 'pong',
            name: 'Pong (RL)',
            description: 'AI agent plays Pong against a heuristic opponent. Both paddles are controlled by simple policies. Watch the RL agent (blue) compete against the heuristic (orange).',
            scenario: true,
            inputs: [
                { id: 'pong_speed', name: 'Game Speed', type: 'number', default: 1.0 }
            ]
        },
        {
            id: 'cnn',
            name: 'CNN Convolution',
            description: 'Visualize how a Convolutional Neural Network applies filter kernels to an input grid. See the convolution operation step by step, producing a feature map.',
            scenario: true,
            inputs: []
        },
        {
            id: 'heat',
            name: 'Heat Equation (PINN)',
            description: 'Physics-Informed Neural Network solving the 2D heat equation. Watch thermal diffusion from a hot center spot as the PDE is solved iteratively.',
            scenario: true,
            inputs: [
                { id: 'heat_alpha', name: 'Diffusion Rate (alpha)', type: 'number', default: 0.2 }
            ]
        },
        {
            id: 'traffic',
            name: 'Traffic Flow (GNN)',
            description: 'Graph Neural Network simulation of traffic flow on a road network. Nodes represent intersections, edges represent roads. Flow diffuses via GNN message passing.',
            scenario: true,
            inputs: [
                { id: 'traffic_diffusion', name: 'Diffusion Rate', type: 'number', default: 0.05 }
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
        // Stop any running scenario
        stopScenarioAnimation();
        scenarioState = null;

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
        });

        if (isScenarioModel(modelId)) {
            // Show scenario canvas, hide network canvas
            let title = currentModel.name;
            showScenarioCard(title);

            // Initialize scenario
            requestAnimationFrame(() => {
                resizeScenarioCanvas();
                if (modelId === 'cartpole') { initCartPole(); drawCartPole(); }
                else if (modelId === 'pong') { initPong(); drawPong(); }
                else if (modelId === 'cnn') { initCNN(); drawCNN(); }
                else if (modelId === 'heat') { initHeat(); drawHeat(); }
                else if (modelId === 'traffic') { initTraffic(); drawTraffic(); }
            });
        } else {
            // Original model: show network canvas, hide scenario card
            hideScenarioCard();
            resizeNetworkCanvas();
        }

        // Reset charts
        renderChart(null);
        renderBenchmark(0);

        // Add listeners to update viz on input change (original models)
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
                        label: 'Inference Time (us)',
                        data: data,
                        backgroundColor: backgroundColors,
                        borderColor: borderColors,
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
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

    // =========================================================================
    // SCENARIO RUN HANDLERS
    // =========================================================================

    function runScenario() {
        if (!currentModel || !isScenarioModel(currentModel.id)) return;

        stopScenarioAnimation();

        const modelId = currentModel.id;

        if (modelId === 'cartpole') {
            // Read params
            const forceEl = document.getElementById('cp_force');
            const gravEl = document.getElementById('cp_gravity');
            initCartPole();
            if (forceEl) scenarioState.forceMag = parseFloat(forceEl.value) || 10;
            if (gravEl) scenarioState.gravity = parseFloat(gravEl.value) || 9.8;
            updateStatus('CartPole: Episode running...');
            outputContent.textContent = 'CartPole RL episode started.\nAgent policy: push in direction of pole lean.';
            runCartPoleAnimation();

        } else if (modelId === 'pong') {
            const speedEl = document.getElementById('pong_speed');
            if (!scenarioState) initPong();
            if (speedEl) scenarioState.speed = parseFloat(speedEl.value) || 1;
            scenarioState.running = true;
            updateStatus('Pong: Game running...');
            outputContent.textContent = 'Pong game started.\nBlue (left) = RL Agent\nOrange (right) = Heuristic opponent';
            runPongAnimation();

        } else if (modelId === 'cnn') {
            if (!scenarioState) initCNN();
            // Cycle to next kernel
            scenarioState.currentKernel = (scenarioState.currentKernel + 1) % scenarioState.kernels.length;
            scenarioState.kernel = scenarioState.kernels[scenarioState.currentKernel];
            // Randomize input
            for (let i = 0; i < scenarioState.inputSize; i++) {
                for (let j = 0; j < scenarioState.inputSize; j++) {
                    scenarioState.inputGrid[i][j] = Math.random();
                }
            }
            recomputeCNN();
            scenarioState.animStep = 0;
            scenarioState.animating = true;
            updateStatus(`CNN: Animating ${scenarioState.kernelNames[scenarioState.currentKernel]} convolution...`);
            outputContent.textContent = `Applying ${scenarioState.kernelNames[scenarioState.currentKernel]} kernel\nInput: ${scenarioState.inputSize}x${scenarioState.inputSize}\nKernel: ${scenarioState.kernelSize}x${scenarioState.kernelSize}\nOutput: ${scenarioState.outputSize}x${scenarioState.outputSize}`;
            animateCNNConvolution();

        } else if (modelId === 'heat') {
            const alphaEl = document.getElementById('heat_alpha');
            if (!scenarioState || scenarioState.step > 200) initHeat();
            if (alphaEl) scenarioState.alpha = parseFloat(alphaEl.value) || 0.2;
            scenarioState.running = true;
            updateStatus('Heat Equation: Diffusion running...');
            outputContent.textContent = 'Heat equation simulation started.\nBoundary conditions: T=0 at edges.\nHot spot initialized at center.';
            runHeatAnimation();

        } else if (modelId === 'traffic') {
            const diffEl = document.getElementById('traffic_diffusion');
            if (!scenarioState) initTraffic();
            if (diffEl) scenarioState.diffusionRate = parseFloat(diffEl.value) || 0.05;
            scenarioState.running = true;
            updateStatus('Traffic GNN: Simulation running...');
            outputContent.textContent = 'Traffic flow simulation started.\nGNN message passing diffuses flow along edges.\nGreen = low traffic, Red = high traffic.';
            runTrafficAnimation();
        }

        // Also try to call the server endpoint (fire and forget, fall back to JS sim)
        fetch(`${API_BASE_URL}/run/${modelId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ inputs: {} }),
            mode: 'cors'
        }).then(r => r.json()).then(data => {
            // If server responds, log it
            outputContent.textContent += '\n\n--- Server response ---\n' + JSON.stringify(data, null, 2);
        }).catch(() => {
            // Server not available - JS simulation handles it
        });
    }

    // =========================================================================
    // BUTTON HANDLERS
    // =========================================================================

    runButton.addEventListener('click', async () => {
        if (!currentModel) {
            return;
        }

        // Handle scenario models
        if (isScenarioModel(currentModel.id)) {
            runScenario();
            return;
        }

        // Original model handling
        const inputs = {};
        currentModel.inputs.forEach((input) => {
            const inputEl = document.getElementById(input.id);
            if (inputEl) {
                inputs[input.id] = inputEl.value;
            }
        });

        updateStatus(`Running Inference on ${currentModel.name}...`);
        outputContent.textContent = 'Running Inference...';

        isRunning = true;
        if (animationId) cancelAnimationFrame(animationId);
        drawNetwork(lastWeights);

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
            updateStatus(`Inference succeeded. Time: ${result.inference_time_us}us`);
            const payload = getVisualizationPayload(result);
            renderChart(payload);
            renderBenchmark(result.inference_time_us);

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

        updateStatus(`Training ${currentModel.name}... (Updating weights)`);
        outputContent.textContent = 'Training...';

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
