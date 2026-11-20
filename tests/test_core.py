"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Prepare a small game dialogue text corpus.
def test_milestone_1():
    from app.core import milestone_1, clean_dialogue, CharTokenizer

    # Test cleaning
    dirty = "  [Hero] Hello!   \r\n\r\n [Guard] Halt!  "
    cleaned = clean_dialogue(dirty)
    assert cleaned == "[Hero] Hello!\n[Guard] Halt!"

    # Test full milestone 1 workflow
    data = milestone_1()
    assert "corpus" in data
    assert "tokenizer" in data
    assert "tokens" in data
    assert "vocab_size" in data

    tokenizer = data["tokenizer"]
    assert isinstance(tokenizer, CharTokenizer)
    assert tokenizer.vocab_size == data["vocab_size"]
    assert tokenizer.vocab_size > 20
    assert len(data["tokens"]) > 100

    # Ensure round-trip encode-decode works cleanly
    sample = "[Hero] Are we ready?"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample


# Milestone 2: Implement a basic GPT architecture and train it on the corpus.
def test_milestone_2():
    import torch
    from app.core import milestone_2, MiniGPT

    result = milestone_2(steps=50, batch_size=4)
    assert "model" in result
    assert "initial_loss" in result
    assert "final_loss" in result
    assert isinstance(result["model"], MiniGPT)

    # Loss should decrease during training
    assert result["final_loss"] < result["initial_loss"]

    # Verify model forward pass shape
    dummy_input = torch.randint(0, result["config"].vocab_size, (2, 16))
    logits, loss = result["model"](dummy_input)
    assert logits.shape == (2, 16, result["config"].vocab_size)
    assert loss is None


# Milestone 3: Generate sample dialogue from the trained GPT.
def test_milestone_3():
    from app.core import milestone_2, milestone_3, generate_dialogue

    m2 = milestone_2(steps=30, batch_size=4)
    res = milestone_3(
        model=m2["model"],
        tokenizer=m2["tokenizer"],
        prompts=["[Hero] ", "[Guard] "],
        max_new_tokens=25,
    )
    assert "generated_samples" in res
    assert "[Hero] " in res["generated_samples"]
    assert "[Guard] " in res["generated_samples"]

    hero_sample = res["generated_samples"]["[Hero] "]
    assert hero_sample.startswith("[Hero] ")
    assert len(hero_sample) > len("[Hero] ")

    # Direct function test with temperature 0 (deterministic)
    det_sample = generate_dialogue(
        m2["model"], m2["tokenizer"], prompt="[Elder] ", max_new_tokens=15, temperature=0.0
    )
    assert det_sample.startswith("[Elder] ")
    assert len(det_sample) > len("[Elder] ")


# Milestone 4: Apply dynamic and static post-training quantization to the model.
def test_milestone_4():
    import torch
    from app.core import milestone_2, milestone_4

    m2 = milestone_2(steps=20, batch_size=4)
    res = milestone_4(model=m2["model"], tokenizer=m2["tokenizer"])

    assert "original_model" in res
    assert "dynamic_model" in res
    assert "static_model" in res

    dummy_input = torch.randint(0, res["original_model"].config.vocab_size, (1, 16))

    out_orig, _ = res["original_model"](dummy_input)
    out_dyn, _ = res["dynamic_model"](dummy_input)
    out_stat, _ = res["static_model"](dummy_input)

    assert out_orig.shape == out_dyn.shape == out_stat.shape
    # Check that dynamic and static models contain quantized operations
    dyn_repr = str(res["dynamic_model"])
    stat_repr = str(res["static_model"])
    assert "DynamicQuantizedLinear" in dyn_repr or "quantized" in dyn_repr
    assert "QuantizedLinear" in stat_repr or "quantized" in stat_repr or "Quantized" in stat_repr


# Milestone 5: Measure and compare the original and quantized model sizes.
def test_milestone_5():
    from app.core import milestone_2, milestone_4, milestone_5

    m2 = milestone_2(steps=20, batch_size=4)
    m4 = milestone_4(model=m2["model"], tokenizer=m2["tokenizer"])
    res = milestone_5(m4)

    assert "original_size_bytes" in res
    assert "dynamic_size_bytes" in res
    assert "static_size_bytes" in res
    assert "reduction_dynamic_pct" in res
    assert "reduction_static_pct" in res

    # Quantized models should have smaller sizes than original FP32 model
    assert res["dynamic_size_bytes"] < res["original_size_bytes"]
    assert res["static_size_bytes"] < res["original_size_bytes"]
    assert res["reduction_dynamic_pct"] > 0
    assert res["reduction_static_pct"] > 0


# Milestone 6: Simulate inference speed by comparing operations on original versus quantized weights.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

