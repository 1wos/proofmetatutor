from services.verifier_api.app.model import LocalVerifierModel


def test_verifier_scores_reasoned_answer_higher() -> None:
    model = LocalVerifierModel()
    strong = model.predict(
        problem_text="Solve 2x + 3 = 11.",
        answer="x = 4",
        explanation="Because subtracting 3 gives 2x = 8, so x = 4.",
    )
    weak = model.predict(
        problem_text="Solve 2x + 3 = 11.",
        answer="x = 4",
        explanation="Maybe it is seven.",
    )

    assert strong["correctness_confidence"] > weak["correctness_confidence"]
    assert "missing_reasoning" in weak["misconception_tags"]

