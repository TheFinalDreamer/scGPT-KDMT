# Examples

## Quick Test

Use the minimal configuration for a fast sanity check:

```bash
python experiments/train_student_kdmt.py \
  --config examples/minimal_config.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/quick_test
```

This runs a small model (2 layers, 32-dim) for 3 epochs to verify the pipeline is working correctly.

## Full Training

For the full experiment, use the standard configuration:

```bash
python experiments/train_student_kdmt.py \
  --config configs/student_kdmt.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/kdmt_student \
  --teacher_checkpoint outputs/teacher/teacher_best.pth
```