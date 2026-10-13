"""Build the frozen JaxNav variants without changing their training loops."""


def build_agent(config):
    if config.get("learner_variant") == "carry-pilot-20260930":
        if config["collector_variant"] == "fixed":
            from .temporal_fixed import temporal_class
            from .consequence_dqn_vectorized import JaxNavConsequenceDQNVectorized as CCE
            from .dqn_vectorized import JaxNavDQNVectorized as DQN
        else:
            from .temporal import temporal_class
            from .consequence_dqn_vectorized_legacy import JaxNavConsequenceDQNVectorized as CCE
            from .dqn_vectorized_legacy import JaxNavDQNVectorized as DQN
        from .carry_pilot import learn_pilot
        cce = config["algorithm"] == "consequence-dqn"
        class Pilot(temporal_class(CCE if cce else DQN, cce)):
            learn = learn_pilot
        return Pilot(config)
    if config.get("learner_variant") == "temporal-20260930":
        from .temporal import temporal_class
        from .consequence_dqn_vectorized_legacy import JaxNavConsequenceDQNVectorized
        from .dqn_vectorized_legacy import JaxNavDQNVectorized
        cce = config["algorithm"] == "consequence-dqn"
        Parent = JaxNavConsequenceDQNVectorized if cce else JaxNavDQNVectorized
        return temporal_class(Parent, cce)(config)
    if config["collector_variant"] == "fixed":
        from .consequence_dqn_vectorized import JaxNavConsequenceDQNVectorized as CCE
        from .dqn_vectorized import JaxNavDQNVectorized as DQN
    elif config["collector_variant"] == "old":
        from .consequence_dqn_vectorized_legacy import JaxNavConsequenceDQNVectorized as CCE
        from .dqn_vectorized_legacy import JaxNavDQNVectorized as DQN
    else:
        raise ValueError("collector_variant must be old or fixed")
    if config["algorithm"] in ("dqn", "dqn-uniform"):
        return DQN(config)
    if config["algorithm"] != "consequence-dqn":
        raise ValueError("Unsupported learner")
    from counterfactual_rl.replay.decay import DecayBuffer, blend_weight

    class ScheduledCCE(CCE):
        replay_buffer_class = DecayBuffer

        def _update(self):
            self.buffer.blend = blend_weight(
                self._current_episode, self.config["decay_enabled"],
                self.config["decay_start"], self.config["decay_end"])
            return super()._update()

    return ScheduledCCE(config)
