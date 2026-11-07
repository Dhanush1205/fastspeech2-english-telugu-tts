# Results Directory

This directory stores evaluation artifacts generated from trained models:

- `test_generated_audios/`: WAV synthesis outputs and extracted prediction `.npz` arrays for assignment test sentences (E1–E4 for English, T1–T4 for Telugu).
- `variance_adaptor_plots/`: Duration, pitch, and energy trajectory plots for test utterances.
- `training_loss.png` / `validation_loss.png`: Training and validation loss curves generated from genuine training runs (`loss_history.npz`).

All artifacts must be produced from actual model execution and genuine checkpoints. No synthetic fallbacks, random trajectories, or placeholder audio are committed.

