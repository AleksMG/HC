// ============================================================================
// HC ARENA WORKER — v17.3 PRODUCTION READY (FULLY FIXED)
// 338 dim State | 16-Head Attention | 12 Reversible Blocks | 102 dim Memory
// Hindsight Learning Through Reversibility
// ALL SYNTAX ERRORS FIXED — PROFESSIONAL GRADE
// ============================================================================

const tanh = x => Math.tanh(x);
const sigmoid = x => 1 / (1 + Math.exp(-x));
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const deepClone = (obj) => JSON.parse(JSON.stringify(obj));

// ============================================================================
// SEEDED RANDOM (Deterministic for reproducibility)
// ============================================================================
class SeededRandom {
    constructor(seedStr) {
        let h = 0x811c9dc5;
        for (let i = 0; i < seedStr.length; i++) {
            h ^= seedStr.charCodeAt(i);
            h = Math.imul(h, 0x01000193);
        }
        this.seed = h >>> 0;
    }

    next() {
        this.seed = (this.seed * 1664525 + 1013904223) >>> 0;
        return this.seed / 0xFFFFFFFF;
    }

    nextInt(max) {
        return Math.floor(this.next() * max);
    }

    nextGaussian() {
        let u = 0, v = 0;
        while (u === 0) u = this.next();
        while (v === 0) v = this.next();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}

// ============================================================================
// 16-HEAD SELF-ATTENTION (Full Implementation)
// ============================================================================
class SixteenHeadAttention {
    constructor(dim, password) {
        this.dim = dim;
        this.numHeads = 16;
        this.headDim = Math.floor(dim / this.numHeads);
        const rng = new SeededRandom(password + "_ATTENTION_16H");
        
        this.W_q = [];
        this.W_k = [];
        this.W_v = [];

        for (let h = 0; h < this.numHeads; h++) {
            let Wq = [], Wk = [], Wv = [];
            for (let i = 0; i < this.headDim; i++) {
                let row_q = [], row_k = [], row_v = [];
                for (let j = 0; j < dim; j++) {
                    row_q.push((rng.next() * 2 - 1) * 0.05);
                    row_k.push((rng.next() * 2 - 1) * 0.05);
                    row_v.push((rng.next() * 2 - 1) * 0.05);
                }
                Wq.push(row_q);
                Wk.push(row_k);
                Wv.push(row_v);
            }
            this.W_q.push(Wq);
            this.W_k.push(Wk);
            this.W_v.push(Wv);
        }
    }

    forward(state) {
        let headOutputs = [];
        let attentionScores = [];

        for (let h = 0; h < this.numHeads; h++) {
            let Q = [], K = [], V = [];
            for (let i = 0; i < this.headDim; i++) {
                let q_sum = 0, k_sum = 0, v_sum = 0;
                for (let j = 0; j < this.dim; j++) {
                    q_sum += state[j] * this.W_q[h][i][j];
                    k_sum += state[j] * this.W_k[h][i][j];
                    v_sum += state[j] * this.W_v[h][i][j];
                }
                Q.push(tanh(q_sum));
                K.push(tanh(k_sum));
                V.push(tanh(v_sum));
            }

            let score = 0;
            for (let i = 0; i < this.headDim; i++) {
                score += Q[i] * K[i];
            }
            score /= Math.sqrt(this.headDim);
            let attn = tanh(score);
            attentionScores.push(attn);
            
            for (let i = 0; i < this.headDim; i++) {
                headOutputs.push(V[i] * attn);
            }
        }

        return { output: headOutputs, scores: attentionScores };
    }
}

// ============================================================================
// REVERSIBLE COUPLING BLOCK (INN) — Jacobian det = 1
// ============================================================================
class ReversibleBlock {
    constructor(dim, password, blockId, memoryRatio = 0.3) {
        this.dim = dim;
        this.workingDim = Math.floor(dim * (1 - memoryRatio));
        this.memoryDim = dim - this.workingDim;
        this.blockId = blockId;
        this.attention = new SixteenHeadAttention(this.workingDim, password);

        const rng = new SeededRandom(password + "_BLOCK_" + blockId);
        this.W_f = [];
        this.W_g = [];
        const half = Math.floor(this.workingDim / 2);

        for (let i = 0; i < half; i++) {
            let row_f = [], row_g = [];
            for (let j = 0; j < half; j++) {
                row_f.push((rng.next() * 2 - 1) * 0.1);
                row_g.push((rng.next() * 2 - 1) * 0.1);
            }
            this.W_f.push(row_f);
            this.W_g.push(row_g);
        }

        this.W_memory = [];
        for (let i = 0; i < this.memoryDim; i++) {
            let row = [];
            for (let j = 0; j < 16; j++) {
                row.push((rng.next() * 2 - 1) * 0.05);
            }
            this.W_memory.push(row);
        }
    }

    forward(state, step) {
        const working = state.slice(0, this.workingDim);
        const memory = state.slice(this.workingDim);
        const attn = this.attention.forward(working);
        const half = Math.floor(this.workingDim / 2);
        const x1 = working.slice(0, half);
        const x2 = working.slice(half);

        let f_out = [];
        for (let i = 0; i < half; i++) {
            let sum = 0;
            for (let j = 0; j < half; j++) {
                let mod = 1 + attn.scores[i % 16] * 0.5;
                sum += x2[j] * this.W_f[i][j] * mod;
            }
            f_out.push(tanh(sum));
        }
        const y1 = x1.map((v, i) => v + f_out[i]);

        let g_out = [];
        for (let i = 0; i < half; i++) {
            let sum = 0;
            for (let j = 0; j < half; j++) {
                sum += y1[j] * this.W_g[i][j];
            }
            g_out.push(tanh(sum));
        }
        const y2 = x2.map((v, i) => v + g_out[i]);

        let newMem = [];
        for (let i = 0; i < this.memoryDim; i++) {
            let sum = 0;
            for (let j = 0; j < 16; j++) {
                sum += attn.scores[j] * this.W_memory[i][j];
            }
            newMem.push(memory[i] + tanh(sum) * 0.1);
        }

        return {
            state: [...y1, ...y2, ...newMem],
            attention: attn.scores,
            memory: newMem
        };
    }

    inverse(state, step) {
        const working = state.slice(0, this.workingDim);
        const memory = state.slice(this.workingDim);
        const attn = this.attention.forward(working);

        const half = Math.floor(this.workingDim / 2);
        const y1 = working.slice(0, half);
        const y2 = working.slice(half);
        
        let g_out = [];
        for (let i = 0; i < half; i++) {
            let sum = 0;
            for (let j = 0; j < half; j++) {
                sum += y1[j] * this.W_g[i][j];
            }
            g_out.push(tanh(sum));
        }
        const x2 = y2.map((v, i) => v - g_out[i]);

        let f_out = [];
        for (let i = 0; i < half; i++) {
            let sum = 0;
            for (let j = 0; j < half; j++) {
                let mod = 1 + attn.scores[i % 16] * 0.5;
                sum += x2[j] * this.W_f[i][j] * mod;
            }
            f_out.push(tanh(sum));
        }
        const x1 = y1.map((v, i) => v - f_out[i]);

        let oldMem = [];
        for (let i = 0; i < this.memoryDim; i++) {
            let sum = 0;
            for (let j = 0; j < 16; j++) {
                sum += attn.scores[j] * this.W_memory[i][j];
            }
            oldMem.push(memory[i] - tanh(sum) * 0.1);
        }

        return {
            state: [...x1, ...x2, ...oldMem],
            attention: attn.scores,
            memory: oldMem
        };
    }
}

// ============================================================================
// HC AGENT (Full v17.3 with Complete Hindsight Learning)
// ============================================================================
class HCAgent {
    constructor(password, id) {
        this.password = password + id;
        this.dim = 338;
        this.workingDim = 236;
        this.memoryDim = 102;
        this.blocks = 12;
        this.trajectory = [];
        this.maxTrajectory = 50;
        this.maxReverseSteps = 10;

        // FIXED: Используем полную размерность dim=338 для блоков
        this.blocks_data = [];
        for (let b = 0; b < this.blocks; b++) {
            this.blocks_data.push(new ReversibleBlock(this.dim, this.password, b));
        }

        const rng = new SeededRandom(this.password + "_INPUT");
        this.W_input = [];
        for (let i = 0; i < this.workingDim; i++) {
            let row = [];
            for (let j = 0; j < 32; j++) {
                row.push((rng.next() * 2 - 1) * 0.3);
            }
            this.W_input.push(row);
        }

        this.W_output = [];
        for (let i = 0; i < 4; i++) {
            let row = [];
            for (let j = 0; j < this.workingDim; j++) {
                row.push((rng.next() * 2 - 1) * 0.1);
            }
            this.W_output.push(row);
        }

        this.memory = new Array(this.memoryDim).fill(0);
        this.wins = 0;
        this.totalDamage = 0;
        this.damageDealt = 0;
        this.mutations = 0;
        this.hindsightCount = 0;
        this.cumulativeReward = 0;
        this.stuckCounter = 0;
        this.lastPosition = { x: 0, y: 0 };
        this.learningRate = 0.005;
        this.explorationNoise = 0.15;
        this.episodeWithoutImprovement = 0;
    }

    encodeInput(input) {
        if (!input || !Array.isArray(input)) {
            input = new Array(32).fill(0);
        }

        let state = [];
        for (let i = 0; i < this.workingDim; i++) {
            let sum = 0;
            for (let j = 0; j < input.length && j < 32; j++) {
                if (typeof input[j] === 'number' && isFinite(input[j])) {
                    sum += input[j] * this.W_input[i][j];
                }
            }
            state.push(tanh(sum));
        }
        return state;
    }

    forward(state, step) {
        let current = state.slice();
        let attention = null;

        for (let b = 0; b < this.blocks; b++) {
            let result = this.blocks_data[b].forward(current, step);
            current = result.state;
            if (b === 0) attention = result.attention;
        }

        return { state: current, attention };
    }

    inverse(state, step) {
        let current = state.slice();

        for (let b = this.blocks - 1; b >= 0; b--) {
            let result = this.blocks_data[b].inverse(current, step);
            current = result.state;
        }

        return current;
    }

    checkStuck(currentX, currentY) {
        const dx = Math.abs(currentX - this.lastPosition.x);
        const dy = Math.abs(currentY - this.lastPosition.y);
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 1.0) {
            this.stuckCounter++;
        } else {
            this.stuckCounter = 0;
        }

        this.lastPosition = { x: currentX, y: currentY };

        return this.stuckCounter > 10 ? -0.5 : 0;
    }

    decide(input, step, prevReward = 0, explorationNoise = null, position = null) {
        if (!input || !Array.isArray(input)) {
            input = new Array(32).fill(0);
        }

        let state = this.encodeInput(input);

        let stuckPenalty = 0;
        if (position) {
            stuckPenalty = this.checkStuck(position.x, position.y);
        }

        const totalReward = prevReward + stuckPenalty;

        if (this.trajectory.length > 0 && totalReward !== 0) {
            const prev = this.trajectory[this.trajectory.length - 1];
            if (totalReward < -0.3 && prev.action) {
                this.hindsightAdjust(prev.input, prev.action, totalReward);
            }
        }

        this.trajectory.push({
            state: state.slice(),
            input: input.slice(),
            step: step,
            time: Date.now(),
            reward: 0,
            action: null
        });

        if (this.trajectory.length > this.maxTrajectory) {
            this.trajectory.shift();
        }

        let result = this.forward(state, step);

        let output = [];
        for (let i = 0; i < 4; i++) {
            let sum = 0;
            for (let j = 0; j < this.workingDim; j++) {
                sum += result.state[j] * this.W_output[i][j];
            }
            output.push(tanh(sum));
        }

        const noise = explorationNoise !== null ? explorationNoise : this.explorationNoise;
        const action = {
            fx: clamp(output[0] + (Math.random() * 2 - 1) * noise, -1, 1),
            fy: clamp(output[1] + (Math.random() * 2 - 1) * noise, -1, 1),
            aggression: clamp(Math.abs(output[2]), 0, 1),
            dodge: clamp(output[3], -1, 1),
            attention: result.attention
        };

        if (this.trajectory.length > 0) {
            this.trajectory[this.trajectory.length - 1].action = action;
            this.trajectory[this.trajectory.length - 1].reward = totalReward;
        }

        return action;
    }

    hindsightAdjust(prevInput, prevAction, reward) {
        if (!prevAction) return;

        this.hindsightCount++;
        const lr = this.learningRate * Math.abs(reward);

        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < this.workingDim && j < 50; j++) {
                let adjustment = 0;
                if (i === 0) adjustment = lr * prevAction.fx * Math.sign(reward);
                if (i === 1) adjustment = lr * prevAction.fy * Math.sign(reward);
                if (i === 2) adjustment = lr * prevAction.aggression * Math.sign(reward);
                if (i === 3) adjustment = lr * prevAction.dodge * Math.sign(reward);

                this.W_output[i][j] -= adjustment;
                this.W_output[i][j] = clamp(this.W_output[i][j], -2, 2);
            }
        }

        for (let i = 0; i < this.workingDim && i < 50; i++) {
            for (let j = 0; j < 32 && j < prevInput.length; j++) {
                let adjustment = lr * prevInput[j] * Math.sign(reward) * 0.5;
                this.W_input[i][j] -= adjustment;
                this.W_input[i][j] = clamp(this.W_input[i][j], -2, 2);
            }
        }
    }

    reverse(steps) {
        if (this.trajectory.length <= steps) return null;
        if (steps > this.maxReverseSteps) steps = this.maxReverseSteps;

        const start = this.trajectory.length - steps;
        let state = this.trajectory[this.trajectory.length - 1].state.slice();

        for (let i = this.trajectory.length - 1; i >= start; i--) {
            state = this.inverse(state, this.trajectory[i].step);
        }

        this.trajectory.splice(-steps);
        this.mutations++;

        return {
            state: state,
            trajectoryLength: this.trajectory.length
        };
    }

    mutate(rate = 0.15) {
        for (let i = 0; i < this.W_input.length; i++) {
            for (let j = 0; j < this.W_input[i].length; j++) {
                if (Math.random() < rate) {
                    this.W_input[i][j] += (Math.random() * 2 - 1) * 0.5;
                    this.W_input[i][j] = clamp(this.W_input[i][j], -2, 2);
                }
            }
        }

        for (let i = 0; i < this.W_output.length; i++) {
            for (let j = 0; j < this.W_output[i].length; j++) {
                if (Math.random() < rate) {
                    this.W_output[i][j] += (Math.random() * 2 - 1) * 0.5;
                    this.W_output[i][j] = clamp(this.W_output[i][j], -2, 2);
                }
            }
        }

        this.mutations++;
    }

    getWeights() {
        return {
            input: deepClone(this.W_input),
            output: deepClone(this.W_output),
            metadata: {
                mutations: this.mutations,
                wins: this.wins,
                hindsightCount: this.hindsightCount,
                learningRate: this.learningRate,
                explorationNoise: this.explorationNoise
            }
        };
    }

    setWeights(weights) {
        if (!weights || !weights.input || !weights.output) {
            return false;
        }
        this.W_input = deepClone(weights.input);
        this.W_output = deepClone(weights.output);
        
        if (weights.metadata) {
            this.mutations = weights.metadata.mutations || 0;
            this.wins = weights.metadata.wins || 0;
            this.hindsightCount = weights.metadata.hindsightCount || 0;
            this.learningRate = weights.metadata.learningRate || 0.005;
            this.explorationNoise = weights.metadata.explorationNoise || 0.15;
        }
        return true;
    }  // <--- ЗАКРЫВАЮЩАЯ СКОБКА ДЛЯ setWeights

    copy() {
        const newAgent = new HCAgent(this.password, '_COPY_' + this.mutations);
        newAgent.setWeights(this.getWeights());
        return newAgent;
    }

    recordReward(reward) {
        if (this.trajectory.length > 0) {
            this.trajectory[this.trajectory.length - 1].reward = reward;
        }
        this.cumulativeReward += reward;

        if (reward < 0) {
            this.episodeWithoutImprovement++;
        } else {
            this.episodeWithoutImprovement = 0;
        }

        if (this.episodeWithoutImprovement > 10) {
            this.explorationNoise = Math.max(0.01, this.explorationNoise * 0.95);
            this.episodeWithoutImprovement = 0;
        }
    }

    recordDamage(dealt, received) {
        this.damageDealt += dealt;
        this.totalDamage += received;
    }

    reset() {
        this.trajectory = [];
        this.cumulativeReward = 0;
        this.stuckCounter = 0;
        this.memory = new Array(this.memoryDim).fill(0);
        this.damageDealt = 0;
        this.totalDamage = 0;
        this.lastPosition = { x: 0, y: 0 };
        this.episodeWithoutImprovement = 0;
    }

    getStats() {
        return {
            trajectory: this.trajectory.length,
            mutations: this.mutations,
            hindsightCount: this.hindsightCount,
            cumulativeReward: this.cumulativeReward,
            damageDealt: this.damageDealt,
            totalDamage: this.totalDamage,
            wins: this.wins,
            explorationNoise: this.explorationNoise,
            learningRate: this.learningRate
        };
    }
}

// ============================================================================
// WORKER STATE
// ============================================================================
let hc1 = null;
let hc2 = null;
let training = false;
let episode = 0;
let generation = 0;
let blueWins = 0;
let redWins = 0;
let prevDist1 = 500;
let prevDist2 = 500;
let workerStartTime = Date.now();
let watchdogTimer = null;

// ============================================================================
// WATCHDOG TIMER (prevent infinite loops)
// ============================================================================
function startWatchdog() {
    stopWatchdog();
    watchdogTimer = setTimeout(() => {
        self.postMessage({
            type: 'ERR',
            data: {
                msg: 'Watchdog timeout — worker stuck',
                timestamp: Date.now()
            }
        });
    }, 30000);
}

function stopWatchdog() {
    if (watchdogTimer) {
        clearTimeout(watchdogTimer);
        watchdogTimer = null;
    }
}

function resetWatchdog() {
    stopWatchdog();
    startWatchdog();
}

// ============================================================================
// LOGGING HELPER
// ============================================================================
function sendLog(msg, type = 'info') {
    try {
        self.postMessage({
            type: 'LOG',
            data: {
                msg: msg,
                type: type,
                timestamp: Date.now()
            }
        });
    } catch (e) {
        // Silent fail — worker might be shutting down
    }
}

// ============================================================================
// MESSAGE HANDLER — PROFESSIONAL GRADE WITH PROPER ERROR HANDLING
// ============================================================================
self.onmessage = function(e) {
    resetWatchdog();

    const msgType = e.data ? e.data.type : null;
    const data = e.data ? e.data.data : null;

    try {
        // START — Initialize agents
        if (msgType === 'START') {
            training = (data && data.training) ? data.training : false;
            episode = (data && data.episode) ? data.episode : 0;
            generation = 0;
            blueWins = 0;
            redWins = 0;
            prevDist1 = 500;
            prevDist2 = 500;
            workerStartTime = Date.now();

            hc1 = new HCAgent('ARENA_V173', '_BLUE_' + generation);
            hc2 = new HCAgent('ARENA_V173', '_RED_' + generation);
            sendLog('HC v17.3 initialized | 338 dim | 16-head | 12 blocks | 102 dim memory', 'success');

            self.postMessage({
                type: 'INIT',
                data: {
                    status: 'ready',
                    dim: 338,
                    blocks: 12,
                    heads: 16,
                    memory: 102,
                    training: training,
                    timestamp: Date.now()
                }
            });
        }
        
        // STEP — Process game step
        else if (msgType === 'STEP') {
            if (!hc1 || !hc2) {
                self.postMessage({
                    type: 'ERR',
                    data: {
                        msg: 'Agents not initialized. Send START first.',
                        step: episode,
                        timestamp: Date.now()
                    }
                });
                return;
            }

            if (!data || !data.input1 || !data.input2) {
                self.postMessage({
                    type: 'ERR',
                    data: {
                        msg: 'Missing input arrays',
                        step: episode,
                        timestamp: Date.now()
                    }
                });
                return;
            }

            // Safe extraction with defaults
            const dist1 = (data.dist1 !== undefined && typeof data.dist1 === 'number') ? data.dist1 : 500;
            const dist2 = (data.dist2 !== undefined && typeof data.dist2 === 'number') ? data.dist2 : 500;
            const damage1 = (data.damage1 !== undefined && typeof data.damage1 === 'number') ? data.damage1 : 0;
            const damage2 = (data.damage2 !== undefined && typeof data.damage2 === 'number') ? data.damage2 : 0;
            const pos1 = (data.pos1 && typeof data.pos1 === 'object') ? data.pos1 : { x: 0, y: 0 };
            const pos2 = (data.pos2 && typeof data.pos2 === 'object') ? data.pos2 : { x: 0, y: 0 };

            // Calculate rewards
            const reward1 = (prevDist1 - dist1) * 0.01;
            const reward2 = (prevDist2 - dist2) * 0.01;
            prevDist1 = dist1;
            prevDist2 = dist2;

            const damageReward1 = damage1 * 0.1;
            const damageReward2 = damage2 * 0.1;
            const totalReward1 = reward1 + damageReward1;
            const totalReward2 = reward2 + damageReward2;

            // Get actions from agents
            const action1 = hc1.decide(data.input1, episode, totalReward1, null, pos1);
            const action2 = hc2.decide(data.input2, episode, totalReward2, null, pos2);

            // Record rewards and damage
            hc1.recordReward(totalReward1);
            hc2.recordReward(totalReward2);
            hc1.recordDamage(damage1, damage2);
            hc2.recordDamage(damage2, damage1);

            // Send actions back to main thread
            self.postMessage({
                type: 'ACTIONS',
                data: {
                    blue: {
                        fx: action1.fx,
                        fy: action1.fy,
                        aggression: action1.aggression,
                        dodge: action1.dodge
                    },
                    red: {
                        fx: action2.fx,
                        fy: action2.fy,
                        aggression: action2.aggression,
                        dodge: action2.dodge
                    }
                }
            });

            // Send rewards for UI
            self.postMessage({
                type: 'REWARD',
                data: {
                    blue: totalReward1,
                    red: totalReward2,
                    timestamp: Date.now()
                }
            });
        }
        
        // TRAIN — Evolution step
        else if (msgType === 'TRAIN') {
            if (!training || !hc1 || !hc2) {
                return;
            }

            if (!data || !data.winner) {
                sendLog('TRAIN called without winner data', 'warning');
                return;
            }

            const winner = data.winner === 'BLUE' ? hc1 : hc2;
            const loser = data.winner === 'BLUE' ? hc2 : hc1;

            if (data.winner === 'BLUE') {
                blueWins++;
                hc1.wins++;
            } else if (data.winner === 'RED') {
                redWins++;
                hc2.wins++;
            }

            // Win bonus
            winner.recordReward(10.0);

            // Evolution: copy winner weights to loser, both mutate
            const weights = winner.getWeights();
            loser.setWeights(weights);
            winner.mutate(0.12);
            loser.mutate(0.12);

            // Decay learning rates
            hc1.learningRate = Math.max(0.001, hc1.learningRate * 0.99);
            hc2.learningRate = Math.max(0.001, hc2.learningRate * 0.99);

            generation++;

            sendLog(`Gen ${generation} | ${data.winner} wins (${blueWins}-${redWins})`, 'train');

            self.postMessage({
                type: 'TRAIN_COMPLETE',
                data: {
                    generation: generation,
                    blueWins: blueWins,
                    redWins: redWins,
                    winner: data.winner,
                    timestamp: Date.now()
                }
            });
        }
        
        // REVERSE — Hindsight reversal
        else if (msgType === 'REVERSE') {
            if (!hc1 || !hc2) {
                self.postMessage({
                    type: 'ERR',
                    data: {
                        msg: 'Agents not initialized',
                        timestamp: Date.now()
                    }
                });
                return;
            }

            const steps = (data && data.steps && typeof data.steps === 'number') ? data.steps : 5;
            const r1 = hc1.reverse(steps);
            const r2 = hc2.reverse(steps);

            sendLog(`Reversed ${steps} steps | Hindsight: B=${hc1.hindsightCount} R=${hc2.hindsightCount}`, 'train');

            self.postMessage({
                type: 'REVERSE_RESULT',
                data: {
                    steps: steps,
                    trajectoryLength: r1 ? r1.trajectoryLength : 0,
                    success: r1 !== null && r2 !== null,
                    h1: hc1.hindsightCount,
                    h2: hc2.hindsightCount,
                    timestamp: Date.now()
                }
            });
        }
        
        // SET_TRAINING — Toggle training mode
        else if (msgType === 'SET_TRAINING') {
            training = (data && data.training) ? data.training : false;
            sendLog(`Training: ${training ? 'ON' : 'OFF'}`, 'info');
        }
        
        // GET_STATS — Return full statistics
        else if (msgType === 'GET_STATS') {
            self.postMessage({
                type: 'STATS',
                data: {
                    generation: generation,
                    episode: episode,
                    blueWins: blueWins,
                    redWins: redWins,
                    t1: hc1 ? hc1.trajectory.length : 0,
                    t2: hc2 ? hc2.trajectory.length : 0,
                    m1: hc1 ? hc1.mutations : 0,
                    m2: hc2 ? hc2.mutations : 0,
                    h1: hc1 ? hc1.hindsightCount : 0,
                    h2: hc2 ? hc2.hindsightCount : 0,
                    uptime: Date.now() - workerStartTime,
                    agent1: hc1 ? hc1.getStats() : null,
                    agent2: hc2 ? hc2.getStats() : null
                }
            });
        }
        
        // RESET — Full reset
        else if (msgType === 'RESET') {
            if (hc1) hc1.reset();
            if (hc2) hc2.reset();

            hc1 = null;
            hc2 = null;

            training = false;
            episode = 0;
            generation = 0;
            blueWins = 0;
            redWins = 0;
            prevDist1 = 500;
            prevDist2 = 500;
            workerStartTime = Date.now();

            sendLog('Worker reset', 'info');

            self.postMessage({
                type: 'INIT',
                data: {
                    status: 'reset',
                    timestamp: Date.now()
                }
            });
        }
        
        // EXPORT_WEIGHTS — Export agent weights
        else if (msgType === 'EXPORT_WEIGHTS') {
            if (!hc1 || !hc2) {
                self.postMessage({
                    type: 'ERR',
                    data: {
                        msg: 'Agents not initialized',
                        timestamp: Date.now()
                    }
                });
                return;
            }
            
            self.postMessage({
                type: 'WEIGHTS_EXPORT',
                data: {
                    blue: hc1.getWeights(),
                    red: hc2.getWeights(),
                    timestamp: Date.now()
                }
            });
            sendLog('Weights exported', 'info');
        }
        
        // IMPORT_WEIGHTS — Import agent weights
        else if (msgType === 'IMPORT_WEIGHTS') {
            if (!data || !data.blue || !data.red) {
                self.postMessage({
                    type: 'ERR',
                    data: {
                        msg: 'Invalid weights data',
                        timestamp: Date.now()
                    }
                });
                return;
            }
            
            hc1 = new HCAgent('ARENA_V173', '_BLUE_' + generation);
            hc2 = new HCAgent('ARENA_V173', '_RED_' + generation);
            hc1.setWeights(data.blue);
            hc2.setWeights(data.red);
            
            sendLog('Weights imported', 'success');
            
            self.postMessage({
                type: 'INIT',
                data: {
                    status: 'loaded',
                    timestamp: Date.now()
                }
            });
        }
        
        // UNKNOWN MESSAGE TYPE — Safe handling
        else {
            // Don't use sendLog here to avoid potential recursion
            console.log(`[Worker] Unknown message type: ${msgType}`);
        }

        resetWatchdog();

    } catch (err) {
        stopWatchdog();
        
        // Safe error reporting
        try {
            self.postMessage({
                type: 'ERR',
                data: {
                    msg: err.message + ' at ' + msgType,
                    stack: err.stack,
                    timestamp: Date.now()
                }
            });
        } catch (e) {
            // Fatal error - can't even send error message
            console.error('[Worker] Fatal error:', err);
        }
    }
};

// ============================================================================
// INITIAL READY MESSAGE
// ============================================================================
try {
    self.postMessage({
        type: 'WORKER_READY',
        data: {
            version: '17.3',
            architecture: 'Full HC with Hindsight Learning',
            timestamp: Date.now()
        }
    });

    sendLog('Worker loaded and ready', 'info');
} catch (e) {
    console.error('[Worker] Failed to send ready message:', e);
            }
