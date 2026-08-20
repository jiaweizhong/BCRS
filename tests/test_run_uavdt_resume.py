from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_uavdt_runner_resumes_incomplete_training_instead_of_trusting_best_pt():
    source = (ROOT / "scripts/esod_baseline/run_uavdt.sh").read_text(encoding="utf-8")

    assert "completed_epochs()" in source
    assert 'results.txt' in source
    assert 'python train.py --resume "$last_ckpt"' in source
    assert 'has only $done_epochs/$EPOCHS completed epochs' in source
    assert 'already trained (found $ckpt), skipping training' not in source
