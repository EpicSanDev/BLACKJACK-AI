# Blackjack AI Advisor

Computer-vision powered Blackjack assistant featuring a synthetic-data training pipeline, a web demo, and a real-time advisor tuned for multi-core CPUs and GPU acceleration (CUDA and Apple Silicon MPS).

## Environment Setup

1. **Python packages**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **PyTorch build**
   - **Apple Silicon (M1/M2)**: `pip install torch torchvision torchaudio` (PyPI wheels support MPS by default).
   - **CUDA GPUs**: follow [PyTorch install selector](https://pytorch.org/get-started/locally/) to match your CUDA version (example: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`).
   - The tooling auto-detects the best device; pass `--device cpu` to force CPU-only execution.

## Training Pipeline

The end-to-end training workflow (data synthesis → split → YOLO fine-tuning) is orchestrated by `model/train_model.py`.

```bash
python model/train_model.py \
  --num-images 4000 \
  --epochs 150 \
  --batch 32 \
  --imgsz 960
```

Key features:
- Synthetic scene generation with realistic card transforms, gradients/backgrounds, and brightness/blur augmentation.
- Automatic train/validation split via `prepare_dataset.split_dataset`.
- Hardware-aware training: dataloader workers default to all CPU cores; GPU selection supports multi-GPU (`0,1,...`) and Apple MPS.
- Configurable CLI flags for datasets, augmentation parameters, optimizer, patience, caching, etc. Use `--help` to view every option.

To reuse an existing dataset without regenerating images, append `--no-regenerate`.

## Real-Time Advisor

`realtime_advisor.py` captures the table region of your monitor and streams YOLO detections into the Blackjack strategy engine.

```bash
python realtime_advisor.py \
  --model runs/detect/blackjack_detector2/weights/best.pt \
  --player-roi 600 800 400 200 \
  --dealer-roi 300 800 200 200 \
  --monitor 1 \
  --view
```

- Crops to the union of player/dealer ROIs (with configurable margin) for higher FPS.
- Advice smoothing reduces flicker; translated outputs (FR) remain on screen.
- Use `--full-screen` if ROIs do not fully cover the playing area.
- Runtime options toggle surrender/double rules to match table specifics.
- Load a learned strategy by passing `--advanced-policy model/advanced_policy.json`; the CLI automatically falls back to chart advice if the policy is missing a state.

## Advanced Strategy Policy

Train a high-accuracy policy from self-play using the new DQN pipeline (default):

```bash
python train_advanced_advisor.py \
  --episodes 800000 \
  --gamma 0.995 \
  --epsilon 1.0 \
  --epsilon-decay 0.999 \
  --hidden-sizes 512 256 128 \
  --device cuda \
  --output model/advanced_policy.json
```

- The DQN setup uses experience replay, target networks, and a multi-layer perceptron (configurable via `--hidden-sizes`) to learn a smooth policy that surpasses the deterministic chart on edge cases (e.g., borderline doubles, marginal surrenders).
- Tweak `--episodes`, `--epsilon-decay`, and network depth to balance training time and convergence. Eight hundred thousand episodes on GPU typically stabilises to sub-0.1% regret versus perfect play.
- Use `--dealer-hits-soft-17`, `--disable-surrender`, or `--disable-double` to replicate the exact table rules you play under. Pass the same options when launching `realtime_advisor.py` so the runtime matches training.
- Split recommendations continue to fall back to the deterministic chart; the learned model governs Hit/Stand/Double/Surrender decisions with the updated ruleset.
- For a lighter baseline (legacy behaviour), rerun with `--algo tabular` to fall back to the original Q-learning table: `python train_advanced_advisor.py --algo tabular --episodes 300000`.
- Once training is done, start the real-time advisor with `--advanced-policy model/advanced_policy.json` to enable the AI-driven recommendations.

## Web Demo

The Flask app (`app/main.py`) provides a lightweight interface for uploading screenshots and receiving annotated advice.

```bash
python app/main.py
```

Visit `http://127.0.0.1:5000/` and upload a table snapshot. The app overlays detections and surfaces:
- Recommended move (translated to French).
- Player total + soft flag.
- Dealer up-card description.

The service shares the same strategy engine as the real-time advisor, ensuring consistent recommendations.

## Repository Layout

```
app/                Flask web interface
blackjack/          Shared strategy utilities (card values, rules, expert tables)
dataset/            Generated data, source assets, and configuration
utils/       Hardware helpers for device selection
model/              Synthetic dataset + training orchestration scripts
realtime_advisor.py Live advisor leveraging screen capture + YOLO
requirements.txt    Minimal dependency set (Ultralytics + runtime libraries)
```

## Next Steps

- Supply additional table background photos in `dataset/backgrounds` to diversify synthetic data.
- Experiment with larger YOLO checkpoints (e.g., `yolov8s.pt`) via `--weights`.
- Calibrate ROIs per casino resolution; save custom presets with shell aliases or wrapper scripts.

Enjoy faster experiments on every CPU core and GPU your machine offers!

## Interactive Blackjack Game

Launch the training-friendly blackjack simulator to generate on-screen hands for the real-time advisor or manual practice:

```bash
python blackjack_game.py --width 1280 --height 720 --use-chart-advisor
```

- Uses the card assets from `dataset/png` and optional table backgrounds under `dataset/backgrounds`.
- Keyboard shortcuts: `Space` = tirer (en jeu) / distribuer (hors jeu), `H` = tirer, `S` = rester, `D` = doubler, `R` = abandonner, flèches haut/bas pour ajuster la mise.
- Buttons mirror the actions; enable `--advanced-policy model/advanced_policy.json` pour charger une policy entraînée.
- The simulator exposes advisor hints (chart ou policy) to validate decisions while furnishing a visual feed for `realtime_advisor.py`.
- Append `--online-learning --use-chart-advisor` to activate on-the-fly Q-learning updates: chaque main jouée alimente un Q-table interne qui affine les conseils en temps réel (`--learning-rate` pour l'alpha, `--exploration` pour l'epsilon-greedy).
