import sys
import os
import json
import tempfile
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

def test_distance_classification_task():
    print("Testing DistanceClassificationTask...")
    from evals.tasks import get_task
    from evals.tasks.distance_classification import DistanceClassificationTask, DistanceClassPromptSpec

    task = get_task("distance_classification", scheme="3-group")
    assert isinstance(task, DistanceClassificationTask)
    spec = task.get_prompt_spec()
    assert isinstance(spec, DistanceClassPromptSpec)
    assert spec.categories == ["A", "B", "C"]

    # Test prompt
    prompt = task.default_prompt()
    assert "FINAL ANSWER: [Letter]" in prompt

    # Test parsing
    assert task.default_parse("FINAL ANSWER: A") == "A"
    assert task.default_parse("The answer is:\nFINAL ANSWER: b") == "B"
    assert task.default_parse("No answer provided") == "UNKNOWN"

    # Test ground truth
    assert task.extract_ground_truth(0.05) == "A"
    assert task.extract_ground_truth(0.2) == "B"
    assert task.extract_ground_truth(1.5) == "C"
    assert task.extract_ground_truth({"Z": 0.05}) == "A"

    print("✓ DistanceClassificationTask tests passed!")

def test_emission_lines_task():
    print("Testing EmissionLineTask...")
    from evals.tasks import get_task
    from evals.tasks.emission_lines import EmissionLineTask, EmissionLinePromptSpec, CANONICAL_LINES

    # Create dummy ground truth df
    dummy_df = pd.DataFrame([
        {"wiki_entity_id": "ent_1", "LINE_NAME": "HALPHA", "SNR": 25.5},
        {"wiki_entity_id": "ent_1", "LINE_NAME": "HALPHA_BROAD", "SNR": 30.0},
        {"wiki_entity_id": "ent_1", "LINE_NAME": "OIII_5007", "SNR": 15.0},
        {"wiki_entity_id": "ent_1", "LINE_NAME": "OIII_4959", "SNR": 8.0},
        {"wiki_entity_id": "ent_1", "LINE_NAME": "HEI_4471", "SNR": 5.0}, # dropped line, should be ignored
        {"wiki_entity_id": "ent_2", "LINE_NAME": "LYALPHA", "SNR": 50.0},
    ])

    task = EmissionLineTask(ground_truth_df=dummy_df)
    assert isinstance(task, EmissionLineTask)
    spec = task.get_prompt_spec()
    assert isinstance(spec, EmissionLinePromptSpec)
    assert len(spec.canonical_lines) == 27
    assert "Hα" in spec.canonical_lines
    assert "[O III] 5007" in spec.canonical_lines

    # Test ground truth extraction and deduplication/max SNR
    gt1 = task.extract_ground_truth("ent_1")
    assert "Hα" in gt1
    assert gt1["Hα"] == 30.0  # Max of HALPHA (25.5) and HALPHA_BROAD (30.0)
    assert "[O III] 5007" in gt1
    assert gt1["[O III] 5007"] == 15.0  # Max of 5007 (15.0) and 4959 (8.0)
    assert "He I 4471" not in gt1  # dropped line not in ground truth

    gt2 = task.extract_ground_truth({"wiki_entity_id": "ent_2"})
    assert gt2 == {"Lyα": 50.0}

    # Test parsing
    # 1. Standard format
    p1 = task.default_parse("Based on the data:\nEMISSION LINES: Hα, [O III] 5007, Hβ")
    assert p1 == ["Hα", "[O III] 5007", "Hβ"]

    # 2. Aliased formats (H-alpha, OIII 5007, etc.)
    p2 = task.default_parse("EMISSION LINES: H-alpha, OIII 5007, Hbeta, Lyman-alpha")
    assert p2 == ["Hα", "[O III] 5007", "Hβ", "Lyα"]

    # 3. Bullets / newlines
    p3 = task.default_parse("EMISSION LINES:\n- H_alpha\n- [N II] 6584\n- SII_6716")
    assert p3 == ["Hα", "[N II] 6583", "[S II] 6720"]

    # 4. NONE
    assert task.default_parse("EMISSION LINES: NONE") == []
    assert task.default_parse("EMISSION LINES: None") == []
    assert task.default_parse("") == []

    print("✓ EmissionLineTask tests passed!")

def test_metrics():
    print("Testing metrics computation...")
    from evals.metrics import compute_and_save_metrics
    from evals.tasks import get_task
    from evals.tasks.emission_lines import EmissionLineTask

    with tempfile.TemporaryDirectory() as tmpdir:
        results_file = os.path.join(tmpdir, "results.jsonl")
        
        # Write test data
        sample_results = [
            {
                "wiki_entity_id": "ent_1",
                "ground_truth_lines": {"Hα": 20.0, "Hβ": 10.0},
                "predicted_lines": ["Hα", "Hβ"],
                "raw_response": "EMISSION LINES: Hα, Hβ",
            },
            {
                "wiki_entity_id": "ent_2",
                "ground_truth_lines": {"Hα": 30.0, "[O III] 5007": 10.0},
                "predicted_lines": ["Hα", "[N II] 6583"], # Missing OIII, added NII
                "raw_response": "EMISSION LINES: Hα, [N II] 6583",
            },
            {
                "wiki_entity_id": "ent_3",
                "ground_truth_lines": {"Lyα": 50.0},
                "predicted_lines": [],
                "raw_response": "EMISSION LINES: NONE",
            }
        ]
        with open(results_file, "w") as f:
            for r in sample_results:
                f.write(json.dumps(r) + "\n")

        task = get_task("emission_lines", ground_truth_df=pd.DataFrame())
        compute_and_save_metrics(tmpdir, task)

        metrics_file = os.path.join(tmpdir, "metrics.json")
        assert os.path.exists(metrics_file)
        with open(metrics_file) as f:
            m = json.load(f)

        assert m["total_samples"] == 3
        assert m["exact_matches"] == 1
        assert "mean_snr_weighted_f1" in m["sample_level"]
        assert "per_line_metrics" in m
        assert "Hα" in m["per_line_metrics"]
        assert m["per_line_metrics"]["Hα"]["tp"] == 2
        assert m["per_line_metrics"]["Hα"]["support"] == 2

    print("✓ Metrics computation tests passed!")

def test_responder_prompt_building():
    print("Testing responder prompt building logic...")
    from evals.tasks import get_task
    from evals.responders.gemini import GeminiResponder
    from evals.responders import ModelResponse

    # Test ModelResponse backward compatibility
    r1 = ModelResponse(parsed="A")
    assert r1.label == "A"
    assert r1.parsed == "A"

    r2 = ModelResponse(label="B")
    assert r2.label == "B"
    assert r2.parsed == "B"

    r3 = ModelResponse(parsed=["Hα", "Hβ"])
    assert r3.parsed == ["Hα", "Hβ"]
    assert r3.label == ""

    # Test prompt generation for both tasks
    dist_task = get_task("distance_classification", scheme="3-group")
    line_task = get_task("emission_lines", ground_truth_df=pd.DataFrame())

    gemini_responder = GeminiResponder.__new__(GeminiResponder)
    p_dist = gemini_responder._build_distance_prompt(dist_task.get_prompt_spec())
    assert "FINAL ANSWER: <label>" in p_dist

    p_lines = gemini_responder._build_emission_prompt(line_task.get_prompt_spec())
    assert "EMISSION LINES: line1, line2" in p_lines
    assert "Hα" in p_lines

    print("✓ Responder prompt building tests passed!")

if __name__ == "__main__":
    test_distance_classification_task()
    test_emission_lines_task()
    test_metrics()
    test_responder_prompt_building()
    print("\n🎉 ALL LOCAL TESTS PASSED!")

