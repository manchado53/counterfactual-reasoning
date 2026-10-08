"""
GardnerChessEnv — single-agent MDP wrapper around pgx Gardner chess.

White (player 0) is the learning agent.
Black (player 1) is the opponent (random legal moves by default;
set opponent='baseline' to use gardner_chess_v0 once haiku compatibility is confirmed).

Key invariant: every state returned to the DQN has current_player == 0 (white to move).
The wrapper executes white's move then the opponent's response before returning,
so the DQN always sees a standard (s, a, r, s') single-agent MDP.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pgx

CHESS_OBS_FLAT = 5 * 5 * 115  # 2875
CHESS_ACTIONS = 1225


class GardnerChessEnv:
    def __init__(self, seed: int = 0, opponent: str = 'random'):
        """
        Args:
            seed:     RNG seed for opponent sampling.
            opponent: 'random' (default) — random legal moves.
                      'baseline' — pgx gardner_chess_v0 AlphaZero model (~1000 Elo).
                      Requires dm-haiku version compatible with current JAX.
        """
        self._env = pgx.make("gardner_chess")
        self.pgx_env = self._env  # expose raw pgx env for use inside lax.scan
        self._rng = jax.random.PRNGKey(seed)
        self._opponent_type = opponent

        if opponent == 'baseline':
            self._opponent_model = pgx.make_baseline_model("gardner_chess_v0")
        else:
            self._opponent_model = None

    def reset(self, key: jax.Array):
        """
        Reset to a new game.

        Returns:
            obs:   (CHESS_OBS_FLAT,) float32 — board from white's perspective
            state: pgx.State with current_player == 0
        """
        state = self._env.init(key)
        return self._obs(state), state

    def step(self, state, action: int):
        """
        Execute white's action, then the opponent's response.

        Args:
            state:  pgx.State with current_player == 0
            action: int — white's move index (must be legal)

        Returns:
            obs:        (CHESS_OBS_FLAT,) float32 — board after both moves
            next_state: pgx.State with current_player == 0
            reward:     float — white's sparse reward (+1 win, -1 loss, 0 otherwise)
            done:       bool
        """
        # --- White's move ---
        s1 = self._env.step(state, jnp.int32(action))
        reward = float(np.array(s1.rewards[0]))
        done = bool(np.array(s1.terminated | s1.truncated))

        if done:
            return self._obs(s1), s1, reward, done

        # --- Opponent's response (current_player == 1 in s1) ---
        self._rng, k = jax.random.split(self._rng)
        black_action = self._opponent_action(s1, k)
        s2 = self._env.step(s1, black_action)

        reward += float(np.array(s2.rewards[0]))
        done = bool(np.array(s2.terminated | s2.truncated))
        return self._obs(s2), s2, reward, done

    def get_legal_mask(self, state) -> np.ndarray:
        """
        Returns (1, CHESS_ACTIONS) bool mask of white's legal moves.

        Shape (1, 1225) matches the (n_agents, actions_per_agent) convention
        used by the DQN training code.
        """
        return np.array(state.legal_action_mask)[np.newaxis, :]

    def _opponent_action(self, state, key):
        """Sample the opponent's action given current state."""
        if self._opponent_type == 'baseline':
            logits, _ = self._opponent_model(state.observation[None])  # (1,5,5,115) → (1,1225)
            masked = jnp.where(state.legal_action_mask, logits[0], -jnp.inf)
            return jax.random.categorical(key, masked)
        else:
            # Random legal move
            legal_indices = jnp.where(
                state.legal_action_mask,
                jnp.arange(CHESS_ACTIONS),
                CHESS_ACTIONS,  # sentinel for illegal
            )
            return jax.random.choice(key, legal_indices,
                                     p=(state.legal_action_mask.astype(jnp.float32) /
                                        state.legal_action_mask.sum()))

    @staticmethod
    def _obs(state) -> np.ndarray:
        """Flatten (5, 5, 115) -> (2875,) float32."""
        return np.array(state.observation, dtype=np.float32).reshape(-1)
