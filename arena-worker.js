// ============ WORKER THREAD (HC Engine + AI) ============

const tanh = x => Math.tanh(x);
const sigmoid = x => 1 / (1 + Math.exp(-x));

// ============ SEEDED RANDOM ============
class SeededRandom {
    constructor(seedStr) {
        let h = 0x811c9dc5;
        for(let i=0; i<seedStr.length; i++) {
            h ^= seedStr.charCodeAt(i);
            h = Math.imul(h, 0x01000193);
        }
        this.seed = h >>> 0;
    }
    next() {
        this.seed = (this.seed * 1664525 + 1013904223) >>> 0;
        return this.seed / 0xFFFFFFFF;
    }
}

// ============ 16-HEAD ATTENTION ============
class SixteenHeadAttention {
    constructor(dim, password) {
        this.dim = dim;
        this.numHeads = 16;
        this.headDim = Math.floor(dim / this.numHeads);
        const rng = new SeededRandom(password + "_ATTN");
        this.W_q = []; this.W_k = []; this.W_v = [];
        for(let h=0; h<this.numHeads; h++) {
            let Wq = [], Wk = [], Wv = [];
            for(let i=0; i<this.headDim; i++) {
                let rq = [], rk = [], rv = [];
                for(let j=0; j<dim; j++) {
                    rq.push((rng.next()*2-1)*0.05);
                    rk.push((rng.next()*2-1)*0.05);
                    rv.push((rng.next()*2-1)*0.05);
                }
                Wq.push(rq); Wk.push(rk); Wv.push(rv);
            }
            this.W_q.push(Wq); this.W_k.push(Wk); this.W_v.push(Wv);
        }
    }
    forward(state) {
        let outputs = [], scores = [];
        for(let h=0; h<this.numHeads; h++) {
            let Q = [], K = [], V = [];
            for(let i=0; i<this.headDim; i++) {
                let qs = 0, ks = 0, vs = 0;
                for(let j=0; j<this.dim; j++) {                    qs += state[j] * this.W_q[h][i][j];
                    ks += state[j] * this.W_k[h][i][j];
                    vs += state[j] * this.W_v[h][i][j];
                }
                Q.push(tanh(qs)); K.push(tanh(ks)); V.push(tanh(vs));
            }
            let score = 0;
            for(let i=0; i<this.headDim; i++) score += Q[i] * K[i];
            score /= Math.sqrt(this.headDim);
            scores.push(tanh(score));
            for(let i=0; i<this.headDim; i++) outputs.push(V[i] * scores[h]);
        }
        return { output: outputs, scores };
    }
}

// ============ REVERSIBLE BLOCK ============
class ReversibleBlock {
    constructor(dim, password, blockId) {
        this.dim = dim;
        this.half = Math.floor(dim / 2);
        const rng = new SeededRandom(password + "_BLOCK_" + blockId);
        this.W_f = []; this.W_g = [];
        for(let i=0; i<this.half; i++) {
            let rf = [], rg = [];
            for(let j=0; j<this.half; j++) {
                rf.push((rng.next()*2-1)*0.1);
                rg.push((rng.next()*2-1)*0.1);
            }
            this.W_f.push(rf); this.W_g.push(rg);
        }
        this.attention = new SixteenHeadAttention(this.half, password);
    }
    
    forward(state) {
        const x1 = state.slice(0, this.half);
        const x2 = state.slice(this.half);
        const attn = this.attention.forward(x1);
        
        let f_out = [];
        for(let i=0; i<this.half; i++) {
            let sum = 0;
            for(let j=0; j<this.half; j++) {
                let mod = 1 + attn.scores[i % 16] * 0.5;
                sum += x2[j] * this.W_f[i][j] * mod;
            }
            f_out.push(tanh(sum));
        }
        const y1 = x1.map((v, i) => v + f_out[i]);
                let g_out = [];
        for(let i=0; i<this.half; i++) {
            let sum = 0;
            for(let j=0; j<this.half; j++) sum += y1[j] * this.W_g[i][j];
            g_out.push(tanh(sum));
        }
        const y2 = x2.map((v, i) => v + g_out[i]);
        
        return { state: [...y1, ...y2], attention: attn.scores };
    }
    
    inverse(state) {
        const y1 = state.slice(0, this.half);
        const y2 = state.slice(this.half);
        const attn = this.attention.forward(y1);
        
        let g_out = [];
        for(let i=0; i<this.half; i++) {
            let sum = 0;
            for(let j=0; j<this.half; j++) sum += y1[j] * this.W_g[i][j];
            g_out.push(tanh(sum));
        }
        const x2 = y2.map((v, i) => v - g_out[i]);
        
        let f_out = [];
        for(let i=0; i<this.half; i++) {
            let sum = 0;
            for(let j=0; j<this.half; j++) {
                let mod = 1 + attn.scores[i % 16] * 0.5;
                sum += x2[j] * this.W_f[i][j] * mod;
            }
            f_out.push(tanh(sum));
        }
        const x1 = y1.map((v, i) => v - f_out[i]);
        
        return { state: [...x1, ...x2], attention: attn.scores };
    }
}

// ============ HC AGENT ============
class HCAgent {
    constructor(password, id) {
        this.password = password + id;
        this.dim = 64;
        this.blocks = 12;
        this.trajectory = [];
        
        this.blocks_data = [];
        for(let b=0; b<this.blocks; b++) {
            this.blocks_data.push(new ReversibleBlock(this.dim, this.password, b));        }
        
        const rng = new SeededRandom(this.password + "_IO");
        this.W_input = [];
        for(let i=0; i<this.dim; i++) {
            let row = [];
            for(let j=0; j<20; j++) row.push((rng.next()*2-1)*0.3);
            this.W_input.push(row);
        }
        
        this.W_output = [];
        for(let i=0; i<4; i++) {
            let row = [];
            for(let j=0; j<this.dim; j++) row.push((rng.next()*2-1)*0.1);
            this.W_output.push(row);
        }
    }
    
    encodeInput(input) {
        let state = [];
        for(let i=0; i<this.dim; i++) {
            let sum = 0;
            for(let j=0; j<input.length && j<20; j++) {
                sum += input[j] * this.W_input[i][j];
            }
            state.push(tanh(sum));
        }
        return state;
    }
    
    forward(state) {
        let current = state.slice();
        let attention = null;
        for(let b=0; b<this.blocks; b++) {
            let result = this.blocks_data[b].forward(current);
            current = result.state;
            if(b === 0) attention = result.attention;
        }
        return { state: current, attention };
    }
    
    inverse(state) {
        let current = state.slice();
        for(let b=this.blocks-1; b>=0; b--) {
            let result = this.blocks_data[b].inverse(current);
            current = result.state;
        }
        return current;
    }
        decide(input, step) {
        let state = this.encodeInput(input);
        this.trajectory.push({ state: state.slice(), step, time: Date.now() });
        if(this.trajectory.length > 100) this.trajectory.shift();
        
        let result = this.forward(state);
        
        let output = [];
        for(let i=0; i<4; i++) {
            let sum = 0;
            for(let j=0; j<this.dim; j++) sum += result.state[j] * this.W_output[i][j];
            output.push(tanh(sum));
        }
        
        return {
            fx: output[0],
            fy: output[1],
            aggression: Math.abs(output[2]),
            dodge: output[3],
            attention: result.attention
        };
    }
    
    reverse(steps) {
        if(this.trajectory.length <= steps) return null;
        const start = this.trajectory.length - steps;
        let state = this.trajectory[this.trajectory.length - 1].state.slice();
        
        for(let i=this.trajectory.length-1; i>=start; i--) {
            state = this.inverse(state);
        }
        
        this.trajectory.splice(-steps);
        return { state: state, trajectoryLength: this.trajectory.length };
    }
    
    mutate(rate=0.15) {
        for(let i=0; i<this.W_input.length; i++) {
            for(let j=0; j<this.W_input[i].length; j++) {
                if(Math.random() < rate) {
                    this.W_input[i][j] += (Math.random() * 2 - 1) * 0.5;
                }
            }
        }
        for(let i=0; i<this.W_output.length; i++) {
            for(let j=0; j<this.W_output[i].length; j++) {
                if(Math.random() < rate) {
                    this.W_output[i][j] += (Math.random() * 2 - 1) * 0.5;
                }
            }        }
    }
    
    getWeights() {
        return { input: this.W_input, output: this.W_output };
    }
    
    setWeights(weights) {
        this.W_input = weights.input;
        this.W_output = weights.output;
    }
}

// ============ WORKER STATE ============
let hc1, hc2;
let training = false;
let episode = 0;
let generation = 0;

// ============ MESSAGE HANDLER ============
self.onmessage = function(e) {
    const { type, data } = e.data;
    
    try {
        if(type === 'START') {
            training = data.training;
            episode = data.episode;
            hc1 = new HCAgent('ARENA_V173', '_BLUE_' + generation);
            hc2 = new HCAgent('ARENA_V173', '_RED_' + generation);
            self.postMessage({ type: 'AGENT_INIT' });
        }
        
        if(type === 'GET_ACTIONS') {
            episode = data.episode;
            training = data.training;
            
            const action1 = hc1.decide(data.input1, episode);
            const action2 = hc2.decide(data.input2, episode);
            
            self.postMessage({
                type: 'ACTIONS',
                data: {
                    blue: action1,
                    red: action2
                }
            });
        }
        
        if(type === 'TRAIN_STEP') {
            if(training) {                const winner = data.winner === 'BLUE' ? hc1 : hc2;
                const loser = data.winner === 'BLUE' ? hc2 : hc1;
                
                // Winner mutates, loser copies
                const weights = winner.getWeights();
                loser.setWeights(JSON.parse(JSON.stringify(weights)));
                winner.mutate(0.12);
                loser.mutate(0.12);
                
                generation++;
                self.postMessage({ type: 'TRAIN_COMPLETE', data: { generation } });
            }
        }
        
        if(type === 'REVERSE') {
            const r1 = hc1.reverse(data.steps);
            const r2 = hc2.reverse(data.steps);
            
            self.postMessage({
                type: 'REVERSE_RESULT',
                data: {
                    steps: data.steps,
                    trajectoryLength: r1 ? r1.trajectoryLength : 0
                }
            });
        }
        
        if(type === 'SET_TRAINING') {
            training = data.training;
        }
    } catch(err) {
        self.postMessage({ type: 'ERROR', data: { message: err.message } });
    }
};

// Initial ready message
self.postMessage({ type: 'WORKER_READY' });
