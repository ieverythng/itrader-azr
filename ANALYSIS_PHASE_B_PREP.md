# iTrader Phase B Preparation Analysis
**Date**: 2025-01-XX  
**Status**: Phase A Complete → Phase B Preparation  
**Focus**: AZR-Aligned Self-Play Trading System

---

## Executive Summary

This analysis reviews the current iTrader implementation against AZR (Absolute Zero Reasoner) principles and identifies optimization opportunities before Phase B implementation. The codebase demonstrates a solid Phase A foundation with a verifiable reward system, task-conditioned policy network, and robust environment design. However, several architectural gaps and optimization opportunities exist for Phase B (proposer-solver self-play).

### Key Findings

✅ **Strengths:**
- Verifiable reward system (PnL-based, environment-executed) aligns with AZR principles
- FiLM-conditioned policy network enables task-aware solving
- Robust reward normalization and component tracking
- Clean separation of concerns (TaskSpec, RewardEngine, TaskEncoder)

⚠️ **Gaps:**
- Task sampling is random (TaskSampler00), not learned (Phase A limitation)
- No proposer model yet (Phase B requirement)
- Task embedding is deterministic hash-based, not learned representations
- Missing self-play loop infrastructure

🔧 **Issues Fixed:**
- `is_flat` type bug (was `float`, now `bool`)

---

## 1. Critical Issues & Fixes

### 1.1 Fixed: Type Error in Reward Function
**File**: `itrader/rl_lib/tasks/rewards.py:75`
- **Issue**: `is_flat` was cast to `float()` instead of `bool()`
- **Impact**: Works but semantically incorrect, could cause confusion
- **Status**: ✅ Fixed

### 1.2 Potential Issue: Task Embedding Normalization
**File**: `itrader/rl_lib/tasks/encoder.py:45-50`
- **Issue**: TaskEncoder normalizes numeric features using `(x - mean) / std`, but this computes statistics over a single 4-element array, making normalization meaningless
- **Impact**: Numeric features may dominate categorical features in embedding
- **Recommendation**: Use fixed scaling ranges instead of per-array normalization, or use learned embeddings (Phase B)

```python
# Current (line 45):
num_vec = (num - num.mean()) / (num.std() + 1e-8)  # mean/std of 4-element array is ~0/1

# Better:
num_vec = num / np.array([1000.0, 5000.0, 10.0, 997.0])  # fixed scale
```

### 1.3 Missing Context: Price Data Window Sampling
**File**: `itrader/envs/itrader_env.py`
- **Observation**: Environment uses fixed dataframe; no mechanism to sample different windows for proposer
- **Phase B Impact**: Proposer needs to select which price window to use as task
- **Recommendation**: Add `start_idx` to TaskSpec for proposer to specify window location

---

## 2. AZR Alignment Analysis

### 2.1 Core AZR Principles vs Current Implementation

| AZR Principle | Current Status | Gap Analysis |
|--------------|----------------|--------------|
| **Verifiable Rewards** | ✅ Excellent | PnL computed from environment execution, fully verifiable |
| **Zero External Data** | ⚠️ Partial | Uses historical OHLCV data (acceptable; market data is "environment") |
| **Self-Play** | ❌ Missing | Phase A: random task sampling, no proposer |
| **Learned Task Generation** | ❌ Missing | TaskSampler00 is random sampling, not learned |
| **Dual Roles (Proposer/Solver)** | ⚠️ Partial | Solver exists; proposer not yet implemented |
| **Outcome-Based Learning** | ✅ Good | Rewards based on PnL outcomes, not imitation |

### 2.2 AZR Reward Design Pattern

**AZR Approach**: Proposer reward = `1 - ((p_solve - 0.5)/0.5)²` where `p_solve` is solver success rate (0-1)

**Current Approach**: Solver gets PnL-based reward; no proposer reward yet

**Alignment Gap**: 
- Need to implement proposer reward function
- Need to track solver success rate per proposed task
- Need `K_rollouts` mechanism to estimate `p_solve`

### 2.3 Task Types in AZR vs iTrader

**AZR**: Three modes (induction, abduction, deduction) on code programs

**iTrader Adaptation**: 
- ✅ **Induction**: Infer market pattern from price history (current solver task)
- ⚠️ **Abduction**: Infer optimal reward params for given window (not yet implemented)
- ⚠️ **Deduction**: Given reward params, predict optimal actions (current solver, but not framed as deduction)

**Recommendation**: Frame Phase B tasks as:
- **Induction**: "Given price window, infer optimal trading strategy"
- **Abduction**: "Given price window and observed outcomes, infer reward parameters that explain behavior"
- **Deduction**: "Given price window and reward params, deduce optimal action sequence"

---

## 3. Architecture Deep Dive

### 3.1 Reward System (`rewards.py`)

**Strengths:**
- Clean registry pattern for extensibility
- Robust NaN/Inf guards via `_f()` helper
- Component decomposition (pnl_delta, turnover, dd, inact) for interpretability
- Normalization via `ret_unit` (scale-invariant rewards)
- Multiple modes (delta vs terminal drawdown)

**AZR Alignment:**
- ✅ Verifiable: all rewards computed from executed trades
- ✅ Outcome-based: no intermediate reasoning supervision
- ⚠️ Missing: proposer-specific reward (will need in Phase B)

**Optimizations:**
1. **Normalization Consistency**: `norm_mode` logic is complex; consider simplifying to two modes: "off" vs "on"
2. **Reward Shaping**: Current design is good; consider adding intrinsic rewards (e.g., diversity of proposed tasks)
3. **Terminal Handling**: Terminal drawdown is computed but rarely used; verify it's properly tested

### 3.2 Task Sampling (`samplers.py`)

**Current**: `TaskSampler00` randomly samples:
- Horizon from discrete set
- Reward weights from ranges
- Price ID (currently "SIM" only)

**Phase B Requirements:**
- Proposer model should generate TaskSpec
- Need buffer of "valid tasks" (similar to AZR's program buffer)
- Proposer should learn which task parameters maximize learnability

**Gap Analysis:**
```python
# Current (random):
horizon = int(self.rng.choice(self.cfg.get("horizons", [96, 128, 160, 200])))

# Phase B (learned):
task_spec = proposer_model.sample_task(context_window)  # returns TaskSpec
```

**Recommendation**: 
1. Keep `TaskSampler00` for Phase A/bootstrap
2. Create `TaskSamplerProposer` for Phase B that wraps proposer model
3. Add task buffer/replay mechanism

### 3.3 Task Embedding (`encoder.py`)

**Current**: Deterministic hash-based embedding
- Categorical fields → SHA256 hash → normalized vector
- Numeric fields → per-array normalization (problematic, see 1.2)

**AZR Approach**: Learned embeddings (not specified in paper, but implied)

**Phase B Recommendation:**
- **Option A**: Keep deterministic for stability, improve numeric scaling
- **Option B**: Learn embeddings via proposer encoder (if using LLM)
- **Option C**: Hybrid: deterministic for Phase A, learnable for Phase B

**Current Issue**: Numeric normalization is meaningless (normalizing 4-element array)

### 3.4 Model Architecture (`solver_mlp_film.py`)

**Design**: FiLM (Feature-wise Linear Modulation) conditioning
- Price features → trunk MLP
- Task embedding → gamma/beta projection
- FiLM: `h_out = gamma * h + beta`

**AZR Alignment**: ✅ Good
- Task-aware policy (critical for multi-task learning)
- Scalable to diverse task distributions

**Strengths:**
- Dropout on task embedding (`t_keep_p`) for regularization
- Clean separation of price vs task features
- Tanh clamp on gamma for stability

**Optimizations:**
1. **Task Dropout**: Current `t_keep_p=0.9` may be too high; consider 0.7-0.8
2. **Task Projection**: `t_proj=64` → `2*hidden` is reasonable; could experiment with wider projection
3. **Architecture Scaling**: For larger models, consider residual connections in trunk

### 3.5 Environment (`itrader_env.py`)

**Design**: Gymnasium-compatible, episode-based trading

**Strengths:**
- Clean state management (`_reset_state()`)
- Per-episode aggregators for callbacks
- Worker-aware seeding for distributed training
- Retry-friendly task override mechanism

**Phase B Gaps:**
1. **Window Selection**: No mechanism for proposer to specify `start_idx` in dataframe
2. **Task Buffer**: No storage/replay of "good tasks" for curriculum learning
3. **Solver Evaluation**: No `K_rollouts` mechanism to evaluate solver on same task multiple times

**Recommendations:**
```python
# Add to TaskSpec:
start_idx: int  # where in dataframe to start episode

# Add to env:
task_buffer: List[TaskSpec]  # buffer of valid tasks (Phase B)

# Add to training loop:
def evaluate_solver(task: TaskSpec, k_rollouts: int = 32) -> float:
    """Return average success rate over K rollouts"""
    successes = []
    for _ in range(k_rollouts):
        _, _, _, _, info = env.reset(task=task)
        # ... run episode ...
        successes.append(info["success"])
    return np.mean(successes)
```

---

## 4. Optimization Opportunities

### 4.1 Low-Level Optimizations

#### A. Reward Function Efficiency
**File**: `rewards.py`
- **Current**: Multiple conditionals, repeated `float()` casts
- **Optimization**: Pre-compute normalization factors, vectorize where possible
- **Impact**: Minor (~5-10% speedup in reward computation)

#### B. Observation Preprocessing
**File**: `itrader_env.py:_get_obs()`
- **Current**: Concatenates deque to array each step
- **Optimization**: Maintain pre-allocated buffer, update incrementally
- **Impact**: Minor (~2-5% speedup)

#### C. Task Embedding Caching
**File**: `itrader_env.py:reset()`
- **Current**: TaskEncoder.embed() called every reset
- **Optimization**: Cache embeddings for identical TaskSpecs
- **Impact**: Minor (only helps if same task appears multiple times)

### 4.2 High-Level Optimizations

#### A. Reward Normalization Consistency
**File**: `rewards.py`
- **Issue**: `norm_mode` has three options ("all", "penalties", "off") but logic is scattered
- **Recommendation**: Refactor to single `normalize` bool + separate penalty scaling
```python
# Proposed:
if normalize:
    pnl_term *= unit
if normalize_penalties:
    penalty_terms *= unit
```

#### B. Task Embedding Improvement
**File**: `encoder.py`
- **Issue**: Numeric normalization is broken (see 1.2)
- **Recommendation**: Use fixed scaling or learnable embeddings
```python
# Fixed scaling:
SCALES = np.array([1000.0, 5000.0, 10.0, 997.0], dtype=np.float32)
num_vec = num / SCALES
num_vec = (num_vec - 0.5) * 2.0  # normalize to [-1, 1] if needed
```

#### C. Episode Aggregation Efficiency
**File**: `itrader_env.py`
- **Current**: Aggregates `rc_sums` every step
- **Optimization**: Only aggregate on episode end (if not needed for callbacks)
- **Impact**: Minor

#### D. Model Architecture Scaling
**File**: `solver_mlp_film.py`
- **Current**: Fixed 2-layer MLP trunk
- **Recommendation**: Make depth configurable, add residual connections for deeper networks
```python
# Configurable depth:
depth = model_config.get("fcnet_depth", 2)
layers = []
for i in range(depth):
    layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
if depth > 2:
    layers.append(nn.BatchNorm1d(hidden))  # optional
```

### 4.3 Code Quality Improvements

#### A. Type Hints
- **Files**: All Python files
- **Status**: Partial (some files have hints, others don't)
- **Recommendation**: Add type hints consistently (helps IDE, catches bugs)

#### B. Documentation
- **Status**: Good docstrings in core files
- **Missing**: Architecture overview, Phase B migration guide
- **Recommendation**: Add `ARCHITECTURE.md` documenting component interactions

#### C. Testing
- **Missing**: Unit tests for reward functions, task encoding
- **Recommendation**: Add tests before Phase B to catch regressions

---

## 5. Phase B Implementation Roadmap

### 5.1 Core Components Needed

#### 1. Proposer Model
**Requirements:**
- Input: Price window (or market context)
- Output: TaskSpec (or parameters to construct TaskSpec)
- Training: RLVR with proposer reward `r_prop = 1 - ((p_solve - 0.5)/0.5)²`

**Architecture Options:**
- **Option A**: Separate policy network (similar to solver, but outputs task params)
- **Option B**: LLM-based proposer (as in `agents/proposer_lora.py` - appears to exist but empty)
- **Option C**: Hybrid: LLM for task generation, policy head for parameter selection

**Recommendation**: Start with Option A (simpler), migrate to Option B if needed

#### 2. Self-Play Loop
**Requirements:**
- Proposer generates batch of tasks
- Solver evaluates each task `K_rollouts` times
- Compute `p_solve` per task → `r_prop`
- Joint training of proposer + solver

**Implementation Sketch:**
```python
def self_play_step(proposer, solver, env, batch_size=64, k_rollouts=32):
    proposals = []
    for _ in range(batch_size):
        # Proposer generates task
        task = proposer.sample_task(context)
        
        # Evaluate solver on task
        success_rate = evaluate_solver(solver, env, task, k_rollouts)
        
        # Compute proposer reward
        r_prop = 1 - ((success_rate - 0.5) / 0.5) ** 2
        
        proposals.append({
            "task": task,
            "logp_prop": proposer.log_prob(task),
            "p_solve": success_rate,
            "r_prop": r_prop,
        })
    
    # Joint training (see train_AZR.py for structure)
    update_proposer(proposals)
    update_solver(proposals)
```

#### 3. Task Buffer (Curriculum Learning)
**Requirements:**
- Store "valid tasks" (tasks that solver can learn from)
- Sample from buffer for proposer context
- Update buffer based on solver performance

**AZR Reference**: Section D.2 (Curriculum Learning via composition)

**Implementation:**
```python
class TaskBuffer:
    def __init__(self, max_size=10000):
        self.tasks: List[TaskSpec] = []
        self.performance: List[float] = []  # solver success rates
        self.max_size = max_size
    
    def add(self, task: TaskSpec, performance: float):
        self.tasks.append(task)
        self.performance.append(performance)
        if len(self.tasks) > self.max_size:
            # Remove worst performers
            ...
    
    def sample_for_context(self, n=10):
        # Sample diverse tasks for proposer context
        ...
```

#### 4. Multi-Task Advantage Estimation
**Requirements:**
- Separate advantages for proposer vs solver
- Normalize per role (as in AZR)
- Joint loss with `lambda_role` weighting

**File**: `train_AZR.py` has skeleton, needs integration with RLlib

### 5.2 Integration with Existing Code

#### A. Extend TaskSpec
```python
@dataclass
class TaskSpec:
    # ... existing fields ...
    start_idx: Optional[int] = None  # for proposer to specify window
    proposed_by: Optional[str] = None  # track proposer version
    p_solve: Optional[float] = None  # cached solver success rate
```

#### B. Extend Environment
```python
def reset(self, *, seed=None, options=None, task: Optional[TaskSpec] = None):
    if task is not None:
        self.current_task = task
        if task.start_idx is not None:
            self.idx = task.start_idx + self.T  # start at specified location
    else:
        self.current_task = self.task_sampler.sample()  # fallback to random
    # ... rest of reset ...
```

#### C. New Training Script
- `train_solo_V2.py` → Keep for Phase A/solver-only
- `train_selfplay.py` → New for Phase B (proposer + solver)

### 5.3 Migration Strategy

**Phase B.1**: Hybrid Mode
- Keep `TaskSampler00` as bootstrap
- Add proposer that can override sampler
- Gradually increase proposer influence (curriculum)

**Phase B.2**: Full Self-Play
- Remove random sampler
- Pure proposer-generated tasks
- Task buffer for curriculum

**Phase B.3**: Advanced Features
- Task composition (like AZR D.2)
- Diversity rewards
- Multi-asset tasks

---

## 6. Recommendations Summary

### Immediate Actions (Before Phase B)

1. ✅ **Fix `is_flat` type bug** (done)
2. 🔧 **Fix TaskEncoder numeric normalization** (see 1.2)
3. 📝 **Add `start_idx` to TaskSpec** (for proposer window selection)
4. 🧪 **Add unit tests** for reward functions and task encoding
5. 📚 **Document architecture** in `ARCHITECTURE.md`

### Phase B Preparation

1. 🏗️ **Implement Proposer Model** (start with policy network)
2. 🔄 **Implement self-play loop** (see `train_AZR.py` skeleton)
3. 📦 **Implement Task Buffer** (curriculum learning)
4. 📊 **Add solver evaluation** (`K_rollouts` mechanism)
5. 🎯 **Implement proposer reward** (`r_prop = 1 - ((p_solve - 0.5)/0.5)²`)

### Optional Optimizations

1. ⚡ **Optimize reward function** (pre-compute, vectorize)
2. 🧠 **Improve task embeddings** (learned or better fixed scaling)
3. 📈 **Scale model architecture** (configurable depth, residuals)
4. 🔍 **Add diversity rewards** (for proposer, see AZR D.4)

---

## 7. Alignment Checklist

### AZR Core Principles

- [x] **Verifiable Rewards**: ✅ PnL from environment execution
- [ ] **Self-Play**: ❌ Not yet implemented (Phase B)
- [ ] **Learned Task Generation**: ❌ Random sampling (Phase B)
- [x] **Outcome-Based Learning**: ✅ No intermediate supervision
- [ ] **Zero External Supervision**: ⚠️ Uses market data (acceptable)
- [ ] **Dual Roles**: ⚠️ Solver exists, proposer missing (Phase B)

### Code Quality

- [x] **Type Safety**: ⚠️ Partial (improve consistency)
- [x] **Documentation**: ✅ Good (add architecture doc)
- [ ] **Testing**: ❌ Missing (add before Phase B)
- [x] **Error Handling**: ✅ Robust (NaN guards, fallbacks)

### Architecture Quality

- [x] **Modularity**: ✅ Excellent (clean separation)
- [x] **Extensibility**: ✅ Good (registry patterns)
- [x] **Scalability**: ✅ Good (distributed-ready, configurable)
- [ ] **Self-Play Ready**: ❌ Missing (Phase B)

---

## 8. Conclusion

The iTrader codebase demonstrates strong foundations for Phase A (solver training) with verifiable rewards, task-conditioned policies, and robust infrastructure. The main gaps for Phase B are the proposer model and self-play loop, which are well-outlined in `train_AZR.py` but need implementation.

**Key Strengths:**
- Verifiable reward system aligns perfectly with AZR
- FiLM-conditioned policy enables multi-task learning
- Clean architecture supports extension to self-play

**Critical Next Steps:**
1. Fix TaskEncoder numeric normalization bug
2. Add `start_idx` to TaskSpec for proposer window selection
3. Implement proposer model and self-play loop
4. Add task buffer for curriculum learning

**Estimated Phase B Timeline:**
- Proposer model: 1-2 weeks
- Self-play loop: 1 week
- Integration & testing: 1 week
- **Total: ~3-4 weeks** (depending on complexity of proposer architecture)

---

**Next Steps**: Begin implementation of proposer model and self-play infrastructure.

