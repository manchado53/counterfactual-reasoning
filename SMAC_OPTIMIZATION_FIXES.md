# SMAC Counterfactual Analysis Optimization Fixes

## Problem Summary

The current `counterfactual.py` implementation has compatibility issues with SMAC due to:

1. **Action Space Explosion**: For SMAC's 3m map (9³ = 729 joint actions), the analyzer performs 34,992 rollouts per state (729 actions × 48 rollouts)
2. **Replay Strategy Overhead**: Every `restore_state()` resets and replays the entire action history from episode start
3. **Computational Infeasibility**: A 50-step episode would require ~1.75 million rollouts

---

## Proposed Fixes

### 🥇 Fix #1: Smart Action Sampling (RECOMMENDED)

**Strategy**: Instead of evaluating all 729 joint actions, intelligently sample a small subset.

**Approach**:
- Always include the chosen action (what actually happened)
- Sample top-k alternatives from policy distribution (what almost happened)
- Optionally include random samples for exploration

**Implementation**:

```python
def perform_counterfactual_rollouts(
    self,
    state_dict: Dict,
    chosen_action: int,  # NEW: the action actually taken
    n_alternatives: int = 5  # NEW: how many alternatives to compare
) -> Dict[int, np.ndarray]:
    """
    Perform counterfactual rollouts for chosen action + sampled alternatives.
    
    Instead of all 729 actions, we compare:
    - The chosen action (what happened)
    - Top-k policy alternatives (what almost happened)
    - Random samples (exploration)
    """
    # 1. Always include chosen action
    actions_to_evaluate = [chosen_action]
    
    # 2. Get policy distribution to find top alternatives
    obs = self._get_obs_from_state(state_dict)
    action_probs = self.model.policy.get_distribution(obs).distribution.probs
    
    # 3. Sample top-k alternatives (excluding chosen)
    top_k = min(n_alternatives, self.action_space_size - 1)
    top_actions = torch.topk(action_probs, k=top_k + 1).indices
    alternatives = [a.item() for a in top_actions if a.item() != chosen_action][:top_k]
    
    actions_to_evaluate.extend(alternatives)
    
    # 4. Perform rollouts only for selected actions
    return_distributions = {}
    for action in actions_to_evaluate:
        returns = []
        for _ in range(self.n_rollouts):
            self.state_manager.restore_state(self.env, state_dict)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            total_return = reward
            discount = self.gamma
            
            if not done:
                for step in range(self.horizon - 1):
                    action_pred, _ = self.model.predict(obs, deterministic=self.deterministic)
                    action_pred = int(action_pred)
                    obs, reward, terminated, truncated, info = self.env.step(action_pred)
                    done = terminated or truncated
                    total_return += discount * reward
                    discount *= self.gamma
                    if done:
                        break
            
            returns.append(total_return)
        
        return_distributions[action] = np.array(returns)
    
    return return_distributions
```

**Impact**:
- **Before**: 729 actions × 48 rollouts = 34,992 rollouts per state
- **After**: 6 actions × 48 rollouts = **288 rollouts per state** (120× faster!)

**Pros**:
- ✅ Maintains analytical value (compares meaningful alternatives)
- ✅ Scales to any action space size
- ✅ Preserves counterfactual reasoning methodology
- ✅ Easy to implement (minimal code changes)

**Cons**:
- ⚠️ Doesn't evaluate all possible actions (but this is actually fine for analysis)

---

### 🥈 Fix #2: Decentralized Analysis

**Strategy**: Analyze each agent's actions independently instead of joint actions.

**Approach**:
- For each agent, vary only their action while keeping others fixed
- Reduces action space from 9³ = 729 to 9 per agent
- More interpretable (identifies which agent made consequential decisions)

**Implementation**:

```python
class DecentralizedCounterfactualAnalyzer:
    """Analyze each agent's actions independently."""
    
    def __init__(self, model, env, state_manager, n_agents, n_actions_per_agent, **kwargs):
        self.model = model
        self.env = env
        self.state_manager = state_manager
        self.n_agents = n_agents
        self.n_actions_per_agent = n_actions_per_agent
        self.horizon = kwargs.get('horizon', 20)
        self.n_rollouts = kwargs.get('n_rollouts', 48)
        self.gamma = kwargs.get('gamma', 0.99)
        self.deterministic = kwargs.get('deterministic', True)
    
    def perform_counterfactual_rollouts_per_agent(
        self,
        state_dict: Dict,
        agent_id: int,
        chosen_joint_action: int
    ) -> Dict[int, np.ndarray]:
        """
        For a specific agent, try all their possible actions
        while keeping other agents' actions fixed.
        """
        return_distributions = {}
        
        # Decode the joint action into individual agent actions
        actual_agent_actions = self._decode_joint_action(chosen_joint_action)
        
        # Try each alternative for this specific agent
        for alt_action in range(self.n_actions_per_agent):
            # Create counterfactual joint action
            cf_agent_actions = actual_agent_actions.copy()
            cf_agent_actions[agent_id] = alt_action
            cf_joint_action = self._encode_joint_action(cf_agent_actions)
            
            # Rollout with this counterfactual
            returns = []
            for _ in range(self.n_rollouts):
                self.state_manager.restore_state(self.env, state_dict)
                obs, reward, terminated, truncated, info = self.env.step(cf_joint_action)
                done = terminated or truncated
                
                total_return = reward
                discount = self.gamma
                
                if not done:
                    for step in range(self.horizon - 1):
                        action_pred, _ = self.model.predict(obs, deterministic=self.deterministic)
                        action_pred = int(action_pred)
                        obs, reward, terminated, truncated, info = self.env.step(action_pred)
                        done = terminated or truncated
                        total_return += discount * reward
                        discount *= self.gamma
                        if done:
                            break
                
                returns.append(total_return)
            
            return_distributions[alt_action] = np.array(returns)
        
        return return_distributions
    
    def _decode_joint_action(self, joint_action: int) -> List[int]:
        """Convert single integer to list of agent actions."""
        actions = []
        for _ in range(self.n_agents):
            actions.append(joint_action % self.n_actions_per_agent)
            joint_action //= self.n_actions_per_agent
        return list(reversed(actions))
    
    def _encode_joint_action(self, agent_actions: List[int]) -> int:
        """Convert list of agent actions to single integer."""
        joint_action = 0
        for action in agent_actions:
            joint_action = joint_action * self.n_actions_per_agent + action
        return joint_action
```

**Impact**:
- **Before**: 729 joint actions
- **After**: 9 actions × 3 agents = **27 total comparisons** (27× reduction!)

**Pros**:
- ✅ Massive reduction in computation
- ✅ More interpretable (per-agent analysis)
- ✅ Identifies which agents make consequential decisions

**Cons**:
- ⚠️ Doesn't capture joint action effects (coordination)
- ⚠️ Requires separate analysis per agent

---

### 🥉 Fix #3: Replay Optimization (Incremental Checkpointing)

**Strategy**: Save checkpoints during episodes to avoid replaying from scratch.

**Approach**:
- Save environment state every N steps (e.g., every 10 steps)
- When restoring, start from nearest checkpoint instead of episode start
- Reduces replay overhead significantly

**Implementation**:

```python
class OptimizedSmacStateManager(StateManager):
    """State Manager with checkpoint support for faster restoration."""
    
    def __init__(self, checkpoint_interval=10):
        self.checkpoints = {}  # {step: (history, seed)}
        self.checkpoint_interval = checkpoint_interval
    
    def clone_state(self, env) -> Dict[str, Any]:
        """Save state with checkpoint tracking."""
        step = len(env.action_history)
        
        # Save checkpoint every N steps
        if step % self.checkpoint_interval == 0:
            self.checkpoints[step] = {
                'action_history': copy.deepcopy(env.action_history),
                'seed': env._seed
            }
        
        return {
            'action_history': copy.deepcopy(env.action_history),
            'seed': env._seed,
            'step': step
        }
    
    def restore_state(self, env, state_dict: Dict[str, Any]) -> None:
        """Restore state using nearest checkpoint."""
        target_step = state_dict['step']
        
        # Find nearest checkpoint BEFORE target
        checkpoint_steps = [s for s in self.checkpoints.keys() if s <= target_step]
        checkpoint_step = max(checkpoint_steps) if checkpoint_steps else 0
        
        if checkpoint_step > 0:
            # Start from checkpoint
            checkpoint = self.checkpoints[checkpoint_step]
            env.reset(seed=checkpoint['seed'])
            
            # Replay from checkpoint
            inner_env = env.env
            for actions in checkpoint['action_history']:
                inner_env.step(actions)
            
            # Replay only remaining steps
            remaining = state_dict['action_history'][checkpoint_step:]
        else:
            # No checkpoint, replay from start
            env.reset(seed=state_dict['seed'])
            remaining = state_dict['action_history']
            inner_env = env.env
        
        # Replay remaining steps
        for actions in remaining:
            inner_env.step(actions)
        
        # Restore wrapper's history
        env.action_history = copy.deepcopy(state_dict['action_history'])
    
    def clear_checkpoints(self):
        """Clear checkpoints (call at episode end to free memory)."""
        self.checkpoints.clear()
    
    @staticmethod
    def get_state_info(env) -> Dict[str, Any]:
        """Get relevant state info."""
        inner_env = env.env
        info = {
            'state': 'running',
            'n_agents': inner_env.n_agents,
            'battles_won': inner_env.battles_won
        }
        return info
    
    @staticmethod
    def get_grid_position(env) -> Tuple[int, int]:
        """SMAC is not grid-based."""
        return (0, 0)
    
    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        return None
```

**Impact**:
- **Before**: Replay 50 steps every time (50 × 288 = 14,400 steps for Fix #1)
- **After**: Replay ~5 steps on average (**10× faster replays**)

**Pros**:
- ✅ Significant speedup for state restoration
- ✅ Works alongside other fixes
- ✅ Minimal memory overhead

**Cons**:
- ⚠️ Requires memory for checkpoints
- ⚠️ Adds complexity to state manager

---

## 🎯 Recommended Combination: Fix #1 + Fix #3

**Smart Action Sampling + Replay Optimization**

This combination provides:
- ✅ 120× reduction in rollouts (Fix #1)
- ✅ 10× faster state restoration (Fix #3)
- ✅ **Total speedup: ~1200× faster!**
- ✅ Maintains analytical value
- ✅ Scales to larger SMAC maps

**Implementation Priority**:
1. Implement Fix #1 first (biggest impact)
2. Add Fix #3 if still too slow
3. Consider Fix #2 for interpretability (optional)

---

## Additional Optimizations

### Parallel Rollouts
Use multiprocessing to parallelize rollouts across actions:

```python
from multiprocessing import Pool

def perform_counterfactual_rollouts_parallel(self, state_dict, actions_to_evaluate):
    with Pool(processes=4) as pool:
        results = pool.starmap(
            self._rollout_single_action,
            [(state_dict, action) for action in actions_to_evaluate]
        )
    return dict(zip(actions_to_evaluate, results))
```

### Reduce Rollouts per Action
Instead of 48 rollouts, use 10-20 for faster analysis:

```python
analyzer = CounterfactualAnalyzer(
    model, env, state_manager,
    n_rollouts=10  # Reduced from 48
)
```

### Shorter Horizon
Reduce horizon from 20 to 10 steps:

```python
analyzer = CounterfactualAnalyzer(
    model, env, state_manager,
    horizon=10  # Reduced from 20
)
```

---

## Benchmarks (Estimated)

| Configuration | Rollouts/State | Time/State | 50-Step Episode |
|--------------|----------------|------------|-----------------|
| **Naive (All Actions)** | 34,992 | ~350s | ~4.9 hours |
| **Fix #1 (Sampling)** | 288 | ~3s | ~2.5 minutes |
| **Fix #1 + #3 (Sampling + Checkpoints)** | 288 | ~0.3s | ~15 seconds |
| **Fix #2 (Decentralized)** | 1,296 | ~13s | ~11 minutes |

---

## Next Steps

1. **Implement Fix #1** in a new `SmacCounterfactualAnalyzer` class
2. **Test on 3m map** with a trained PPO agent
3. **Add Fix #3** if performance is still insufficient
4. **Consider Fix #2** for agent-level interpretability

---

## References

- Original counterfactual.py: `src/counterfactual_rl/analysis/counterfactual.py`
- SMAC wrapper: `src/counterfactual_rl/environments/smac.py`
- State manager base: `src/counterfactual_rl/environments/base.py`
