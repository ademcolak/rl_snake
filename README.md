# Snake Game with Reinforcement Learning

A modular implementation of the classic Snake game with reinforcement learning using Gymnasium and Stable Baselines3.

## Project Structure

```
├── snake_game.py          # Core game logic + manual play
├── snake_env.py           # Gymnasium environment wrapper
├── train_snake.py         # Training script with SB3
├── test_improvements.py   # Quick testing script
├── pyproject.toml         # Project configuration and dependencies
└── README.md              # This file
```

## Installation with uv

1. Initialize the project:
```bash
uv init
```

2. Install dependencies:
```bash
uv sync
```

## Usage

### Play Snake Manually

Before training the AI, you can play the game yourself:

```bash
# Basic play
uv run python snake_game.py

# Custom settings
uv run python snake_game.py --width 25 --height 20 --speed 10
```

**Manual Controls:**
- **Arrow Keys**: Move snake (Up, Right, Down, Left)
- **SPACE**: Pause/Unpause game
- **R**: Restart game
- **ESC**: Quit game

### Quick Test (Recommended)

Run a quick training and evaluation test (50k steps, ~5-10 minutes):
```bash
uv run python test_improvements.py
```

Or with custom timesteps:
```bash
uv run python test_improvements.py 200000
```

### Basic Training

Train with default settings (PPO, 100k steps):
```bash
uv run python train_snake.py
```

### Advanced Training Options

Train for more steps:
```bash
uv run python train_snake.py train 200000
```

Train with visual rendering (slower but you can watch):
```bash
uv run python train_snake.py train 50000 --render
```

### Watch Trained Model

Watch the AI play after training:
```bash
uv run python train_snake.py play
```

Watch specific model play 10 games:
```bash
uv run python train_snake.py play snake_model 10
```

## Environment Details

### State Space (33 dimensions)
- **Local Grid Vision (25)**: 5x5 grid around snake head
  - 0 = Empty space
  - 1 = Wall/boundary
  - 2 = Snake body
  - 3 = Food
- **Direction (4)**: Current direction (up, right, down, left) - one-hot encoded
- **Food Direction (4)**: Global food location relative to head (up, down, left, right)

### Action Space (3 actions - Relative Turns)
- **0**: Continue straight (no turn)
- **1**: Turn right (clockwise 90°)
- **2**: Turn left (counter-clockwise 90°)

The snake uses relative actions instead of absolute directions because:
- More intuitive for the agent to learn (snake can't reverse direction)
- Reduces action space complexity
- All possible moves are covered with just 3 actions

### Reward System
- **+10**: Eating food
- **-10**: Game over (collision)
- **+0.1**: Moving closer to food (distance-based shaping)
- **-0.1**: Moving away from food (distance-based shaping)

## Files Explained

### `snake_game.py`
Pure Python implementation of Snake game logic:
- No RL dependencies
- Handles game state, collisions, scoring
- Pygame rendering support
- Clean separation of game logic from AI

### `snake_env.py`
Gymnasium environment wrapper:
- Implements standard Gym interface
- Handles observation/action space definitions
- Manages rendering modes
- Environment registration

### `train_snake.py`
Complete training pipeline:
- Algorithm: PPO (Proximal Policy Optimization)
- Neural Network: 3-layer architecture [256, 256, 128] with ReLU
- Entropy coefficient: 0.01 for better exploration
- Training progress monitoring
- Model saving/loading
- Command-line interface

### `test_improvements.py`
Quick testing and evaluation script:
- Trains model with specified timesteps (default: 50k)
- Automatically evaluates performance over 10 episodes
- Shows expected vs actual performance improvements

## Expected Performance

With the improved state representation, reward shaping, and deeper neural network:

- **Random Agent**: ~0-2 score
- **Trained Agent (50k steps)**: ~15-25+ score (previously 5-15)
- **Well-trained Agent (200k+ steps)**: ~30-50+ score (previously 15-30)

### Key Improvements
1. **5x5 Local Vision**: Snake can see 2 squares ahead in all directions
2. **Distance-based Rewards**: Immediate feedback on every move
3. **Deeper Network**: 3-layer [256, 256, 128] architecture
4. **Better Exploration**: Entropy coefficient prevents premature convergence

