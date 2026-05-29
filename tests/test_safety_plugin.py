from agents.tutor_agent.safety_plugin import SafetyPlugin


def test_safety_blocks_personal_label() -> None:
    plugin = SafetyPlugin()
    decision = plugin.after_model_callback(
        model_output="You are bad at math.",
        requires_reasoning_first=False,
    )

    assert not decision.allowed
    assert decision.reason == "personal_label"

