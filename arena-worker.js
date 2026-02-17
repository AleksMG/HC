// ============================================================================
// HC ARENA WORKER — FULL v17.3 WITH HINDSIGHT LEARNING
// 338 dim | 16-Head Attention | 12 Reversible Blocks | 102 dim Memory
// ============================================================================

const tanh = x => Math.tanh(x);
const sigmoid = x => 1 / (1 + Math.exp(-x));
const mod = (n, m) => ((n % m) + m) % m;
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

// ============================================================================
// SEEDED RANDOM
// ============================================================================
class SeededRandom {
    constructor(seedStr) {
        let h = 0x811c9dc5;
        for(let i = 0; i < seedStr.length; i++) {
            h ^= seedStr.charCodeAt(i);
            h = Math.imul(h, 0x01000193);
        }
        this.seed = h >>> 0;
    }
    next() {
        this.seed = (this.seed * 1664525 + 1013904223) >>> 0;
        return this.seed / 0xFFFFFFFF;
    }
    nextInt(max) { return Math.floor(this.next() * max); }
    nextGaussian() {
        let u = 0, v = 0;
        while(u === 0) u = this.next();
        while(v === 0) v = this.next();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}

// ============================================================================
// 16-HEAD SELF-ATTENTION
// ============================================================================
class SixteenHeadAttention {
    constructor(dim, password) {
        this.dim = dim;
        this.numHeads = 16;
        this.headDim = Math.floor(dim / this.numHeads);
        const rng = new SeededRandom(password + "_ATTENTION_16H");
        this.W_q = []; this.W_k = []; this.W_v = [];
        for(let h = 0; h < this.numHeads; h++) {
            let Wq = [], Wk = [], Wv = [];
            for(let i = 0; i < this.headDim; i++) {
                let row_q = [], row_k = [], row_v = [];
                for(let j = 0; j < dim; j++) {                    row_q.push((rng.next() * 2 - 1) * 0.05);
                    row_k.push((rng.next() * 2 - 1) * 0.05);
                    row_v.push((rng.next() * 2 - 1) * 0.05);
                }
                Wq.push(row_q); Wk.push(row_k); Wv.push(row_v);
            }
            this.W_q.push(Wq); this.W_k.push(Wk); this.W_v.push(Wv);
        }
    }
    forward(state) {
        let headOutputs = [], attentionScores = [];
        for(let h = 0; h < this.numHeads; h++) {
            let Q = [], K = [], V = [];
            for(let i = 0; i < this.headDim; i++) {
                let q_sum = 0, k_sum = 0, v_sum = 0;
                for(let j = 0; j < this.dim; j++) {
                    q_sum += state[j] * this.W_q[h][i][j];
                    k_sum += state[j] * this.W_k[h][i][j];
                    v_sum += state[j] * this.W_v[h][i][j];
                }
                Q.push(tanh(q_sum)); K.push(tanh(k_sum)); V.push(tanh(v_sum));
            }
            let score = 0;
            for(let i = 0; i < this.headDim; i++) score += Q[i] * K[i];
            score /= Math.sqrt(this.headDim);
            let attn = tanh(score);
            attentionScores.push(attn);
            for(let i = 0; i < this.headDim; i++) headOutputs.push(V[i] * attn);
        }
        return { output: headOutputs, scores: attentionScores };
    }
}

// ============================================================================
// REVERSIBLE COUPLING BLOCK (INN)
// ============================================================================
class ReversibleBlock {
    constructor(dim, password, blockId, memoryRatio = 0.3) {
        this.dim = dim;
        this.workingDim = Math.floor(dim * (1 - memoryRatio));
        this.memoryDim = dim - this.workingDim;
        this.blockId = blockId;
        this.attention = new SixteenHeadAttention(this.workingDim, password);
        const rng = new SeededRandom(password + "_BLOCK_" + blockId);
        this.W_f = []; this.W_g = [];
        const half = Math.floor(this.workingDim / 2);
        for(let i = 0; i < half; i++) {
            let row_f = [], row_g = [];
            for(let j = 0; j < half; j++) {
                row_f.push((rng.next() * 2 - 1) * 0.1);                row_g.push((rng.next() * 2 - 1) * 0.1);
            }
            this.W_f.push(row_f); this.W_g.push(row_g);
        }
        this.W_memory = [];
        for(let i = 0; i < this.memoryDim; i++) {
            let row = [];
            for(let j = 0; j < 16; j++) row.push((rng.next() * 2 - 1) * 0.05);
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
        for(let i = 0; i < half; i++) {
            let sum = 0;
            for(let j = 0; j < half; j++) {
                let mod = 1 + attn.scores[i % 16] * 0.5;
                sum += x2[j] * this.W_f[i][j] * mod;
            }
            f_out.push(tanh(sum));
        }
        const y1 = x1.map((v, i) => v + f_out[i]);
        let g_out = [];
        for(let i = 0; i < half; i++) {
            let sum = 0;
            for(let j = 0; j < half; j++) sum += y1[j] * this.W_g[i][j];
            g_out.push(tanh(sum));
        }
        const y2 = x2.map((v, i) => v + g_out[i]);
        let newMem = [];
        for(let i = 0; i < this.memoryDim; i++) {
            let sum = 0;
            for(let j = 0; j < 16; j++) sum += attn.scores[j] * this.W_memory[i][j];
            newMem.push(memory[i] + tanh(sum) * 0.1);
        }
        return { state: [...y1, ...y2, ...newMem], attention: attn.scores, memory: newMem };
    }
    inverse(state, step) {
        const working = state.slice(0, this.workingDim);
        const memory = state.slice(this.workingDim);
        const attn = this.attention.forward(working);
        const half = Math.floor(this.workingDim / 2);
        const y1 = working.slice(0, half);
        const y2 = working.slice(half);        let g_out = [];
        for(let i = 0; i < half; i++) {
            let sum = 0;
            for(let j = 0; j < half; j++) sum += y1[j] * this.W_g[i][j];
            g_out.push(tanh(sum));
        }
        const x2 = y2.map((v, i) => v - g_out[i]);
        let f_out = [];
        for(let i = 0; i < half; i++) {
            let sum = 0;
            for(let j = 0; j < half; j++) {
                let mod = 1 + attn.scores[i % 16] * 0.5;
                sum += x2[j] * this.W_f[i][j] * mod;
            }
            f_out.push(tanh(sum));
        }
        const x1 = y1.map((v, i) => v - f_out[i]);
        let oldMem = [];
        for(let i = 0; i < this.memoryDim; i++) {
            let sum = 0;
            for(let j = 0; j < 16; j++) sum += attn.scores[j] * this.W_memory[i][j];
            oldMem.push(memory[i] - tanh(sum) * 0.1);
        }
        return { state: [...x1, ...x2, ...oldMem], attention: attn.scores, memory: oldMem };
    }
}

// ============================================================================
// HC AGENT (FULL v17.3 WITH HINDSIGHT LEARNING)
// ============================================================================
class HCAgent {
    constructor(password, id) {
        this.password = password + id;
        this.dim = 338;
        this.workingDim = 236;
        this.memoryDim = 102;
        this.blocks = 12;
        this.trajectory = [];
        this.blocks_data = [];
        for(let b = 0; b < this.blocks; b++) {
            this.blocks_data.push(new ReversibleBlock(this.workingDim, this.password, b));
        }
        const rng = new SeededRandom(this.password + "_INPUT");
        this.W_input = [];
        for(let i = 0; i < this.workingDim; i++) {
            let row = [];
            for(let j = 0; j < 32; j++) row.push((rng.next() * 2 - 1) * 0.3);
            this.W_input.push(row);
        }
        this.W_output = [];        for(let i = 0; i < 4; i++) {
            let row = [];
            for(let j = 0; j < this.workingDim; j++) row.push((rng.next() * 2 - 1) * 0.1);
            this.W_output.push(row);
        }
        this.memory = new Array(this.memoryDim).fill(0);
        this.wins = 0;
        this.totalDamage = 0;
        this.mutations = 0;
        this.hindsightCount = 0;
        this.cumulativeReward = 0;
    }

    encodeInput(input) {
        let state = [];
        for(let i = 0; i < this.workingDim; i++) {
            let sum = 0;
            for(let j = 0; j < input.length && j < 32; j++) {
                sum += input[j] * this.W_input[i][j];
            }
            state.push(tanh(sum));
        }
        return state;
    }

    forward(state, step) {
        let current = state.slice();
        let attention = null;
        for(let b = 0; b < this.blocks; b++) {
            let result = this.blocks_data[b].forward(current, step);
            current = result.state;
            if(b === 0) attention = result.attention;
        }
        return { state: current, attention };
    }

    inverse(state, step) {
        let current = state.slice();
        for(let b = this.blocks - 1; b >= 0; b--) {
            let result = this.blocks_data[b].inverse(current, step);
            current = result.state;
        }
        return current;
    }

    decide(input, step, prevReward = 0) {
        let state = this.encodeInput(input);
        
        if(this.trajectory.length > 0 && prevReward !== 0) {
            const prev = this.trajectory[this.trajectory.length - 1];            if(prevReward < -0.5) {
                this.hindsightAdjust(prev.input, prev.action, prevReward);
            }
        }
        
        this.trajectory.push({ 
            state: state.slice(), 
            input: input.slice(),
            step, 
            time: Date.now(),
            reward: 0,
            action: null
        });
        if(this.trajectory.length > 200) this.trajectory.shift();
        
        let result = this.forward(state, step);
        let output = [];
        for(let i = 0; i < 4; i++) {
            let sum = 0;
            for(let j = 0; j < this.workingDim; j++) sum += result.state[j] * this.W_output[i][j];
            output.push(tanh(sum));
        }
        
        const action = {
            fx: output[0],
            fy: output[1],
            aggression: Math.abs(output[2]),
            dodge: output[3],
            attention: result.attention
        };
        
        if(this.trajectory.length > 0) {
            this.trajectory[this.trajectory.length - 1].action = action;
        }
        
        return action;
    }

    hindsightAdjust(prevInput, prevAction, reward) {
        this.hindsightCount++;
        const lr = 0.01 * Math.abs(reward);
        for(let i = 0; i < 4; i++) {
            for(let j = 0; j < this.workingDim && j < 50; j++) {
                if(i === 0) this.W_output[i][j] -= lr * prevAction.fx * Math.sign(reward);
                if(i === 1) this.W_output[i][j] -= lr * prevAction.fy * Math.sign(reward);
            }
        }
        for(let i = 0; i < 4; i++) {
            for(let j = 0; j < this.workingDim; j++) {
                this.W_output[i][j] = clamp(this.W_output[i][j], -2, 2);            }
        }
    }

    reverse(steps) {
        if(this.trajectory.length <= steps) return null;
        const start = this.trajectory.length - steps;
        let state = this.trajectory[this.trajectory.length - 1].state.slice();
        for(let i = this.trajectory.length - 1; i >= start; i--) {
            state = this.inverse(state, this.trajectory[i].step);
        }
        this.trajectory.splice(-steps);
        this.mutations++;
        return { state, trajectoryLength: this.trajectory.length };
    }

    mutate(rate = 0.15) {
        for(let i = 0; i < this.W_input.length; i++) {
            for(let j = 0; j < this.W_input[i].length; j++) {
                if(Math.random() < rate) {
                    this.W_input[i][j] += (Math.random() * 2 - 1) * 0.5;
                    this.W_input[i][j] = clamp(this.W_input[i][j], -2, 2);
                }
            }
        }
        for(let i = 0; i < this.W_output.length; i++) {
            for(let j = 0; j < this.W_output[i].length; j++) {
                if(Math.random() < rate) {
                    this.W_output[i][j] += (Math.random() * 2 - 1) * 0.5;
                    this.W_output[i][j] = clamp(this.W_output[i][j], -2, 2);
                }
            }
        }
        this.mutations++;
    }

    getWeights() {
        return { input: JSON.parse(JSON.stringify(this.W_input)), output: JSON.parse(JSON.stringify(this.W_output)) };
    }

    setWeights(weights) {
        this.W_input = JSON.parse(JSON.stringify(weights.input));
        this.W_output = JSON.parse(JSON.stringify(weights.output));
    }

    copy() {
        const newAgent = new HCAgent(this.password, '_COPY_' + this.mutations);
        newAgent.setWeights(this.getWeights());
        newAgent.mutations = this.mutations;
        newAgent.wins = this.wins;        return newAgent;
    }

    recordReward(reward) {
        if(this.trajectory.length > 0) {
            this.trajectory[this.trajectory.length - 1].reward = reward;
        }
        this.cumulativeReward += reward;
    }
}

// ============================================================================
// WORKER STATE
// ============================================================================
let hc1 = null, hc2 = null;
let training = false, episode = 0, generation = 0;
let blueWins = 0, redWins = 0;
let prevDist1 = 500, prevDist2 = 500;

// ============================================================================
// MESSAGE HANDLER
// ============================================================================
self.onmessage = function(e) {
    const { type, data } = e.data;
    try {
        if(type === 'START') {
            training = data.training || false;
            episode = data.episode || 0;
            generation = 0; blueWins = 0; redWins = 0;
            hc1 = new HCAgent('ARENA_V173', '_BLUE_' + generation);
            hc2 = new HCAgent('ARENA_V173', '_RED_' + generation);
            prevDist1 = 500; prevDist2 = 500;
            self.postMessage({ type: 'INIT', data: { status: 'ready', dim: 338, blocks: 12, heads: 16, memory: 102 } });
            self.postMessage({ type: 'LOG', data: { msg: 'HC v17.3 initialized | 338 dim | 16-head | 12 blocks | 102 dim memory' } });
        }

        if(type === 'STEP') {
            episode = data.episode || episode;
            training = data.training !== undefined ? data.training : training;
            if(!hc1 || !hc2) { 
                self.postMessage({ type: 'ERR', data: { msg: 'Agents not initialized' } }); 
                return; 
            }
            
            const dist1 = data.dist1 || 500, dist2 = data.dist2 || 500;
            const reward1 = (prevDist1 - dist1) * 0.01;
            const reward2 = (prevDist2 - dist2) * 0.01;
            prevDist1 = dist1; prevDist2 = dist2;
            
            const action1 = hc1.decide(data.input1, episode, reward1);            const action2 = hc2.decide(data.input2, episode, reward2);
            
            hc1.recordReward(reward1);
            hc2.recordReward(reward2);
            
            self.postMessage({
                type: 'ACTIONS',
                data: {
                    blue: { fx: action1.fx, fy: action1.fy, aggression: action1.aggression, dodge: action1.dodge },
                    red: { fx: action2.fx, fy: action2.fy, aggression: action2.aggression, dodge: action2.dodge }
                }
            });
        }

        if(type === 'TRAIN') {
            if(training && hc1 && hc2) {
                const winner = data.winner === 'BLUE' ? hc1 : hc2;
                const loser = data.winner === 'BLUE' ? hc2 : hc1;
                if(data.winner === 'BLUE') blueWins++; else redWins++;
                
                const weights = winner.getWeights();
                loser.setWeights(weights);
                winner.mutate(0.12);
                loser.mutate(0.12);
                
                generation++;
                self.postMessage({ type: 'TRAIN_COMPLETE', data: { generation, blueWins, redWins, winner: data.winner } });
                self.postMessage({ type: 'LOG', data: { msg: 'Gen ' + generation + ' | ' + data.winner + ' wins (' + blueWins + '-' + redWins + ')' } });
            }
        }

        if(type === 'REVERSE') {
            if(!hc1 || !hc2) { 
                self.postMessage({ type: 'ERR', data: { msg: 'Agents not initialized' } }); 
                return; 
            }
            const steps = data.steps || 5;
            const r1 = hc1.reverse(steps);
            const r2 = hc2.reverse(steps);
            self.postMessage({ 
                type: 'REVERSE_RESULT', 
                data: { steps, trajectoryLength: r1 ? r1.trajectoryLength : 0, success: r1 && r2, h1: hc1.hindsightCount, h2: hc2.hindsightCount } 
            });
            self.postMessage({ type: 'LOG', data: { msg: 'Reversed ' + steps + ' steps | Hindsight: B=' + hc1.hindsightCount + ' R=' + hc2.hindsightCount } });
        }

        if(type === 'SET_TRAINING') { training = data.training; }
        
        if(type === 'RESET') {
            hc1 = null; hc2 = null; training = false; episode = 0; generation = 0; blueWins = 0; redWins = 0;            self.postMessage({ type: 'INIT', data: { status: 'reset' } });
        }

    } catch(err) { 
        self.postMessage({ type: 'ERR', data: { msg: err.message + ' at ' + type } }); 
    }
};

self.postMessage({ type: 'WORKER_READY', data: { version: '17.3', architecture: 'Full HC with Hindsight Learning' } });
