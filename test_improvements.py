#!/usr/bin/env python3

import sys
from train_snake import train_snake, play_trained_model

def quick_test(timesteps=50000):
    """Quick training test with new improvements"""

    print("=" * 60)
    print("TESTING IMPROVED RL SNAKE")
    print("=" * 60)
    print()
    print("Improvements:")
    print("  1. State: 11 → 29 dimensions (5x5 local vision)")
    print("  2. Reward: Distance-based shaping added")
    print("  3. Network: [64,64] → [256,256,128] (deeper)")
    print("  4. Entropy: 0.01 for better exploration")
    print()
    print("=" * 60)
    print()

    print(f"Training for {timesteps} timesteps...")
    print("This will take a few minutes...")
    print()

    model = train_snake(timesteps=timesteps, render=False)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE - Testing model performance")
    print("=" * 60)
    print()

    scores = play_trained_model(model_path="snake_model", episodes=10)

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Episodes tested: 10")
    print(f"Average score: {sum(scores)/len(scores):.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Worst score: {min(scores)}")
    print()
    print("Expected performance:")
    print("  - Random agent: 0-2")
    print("  - Old model (50k): 5-15")
    print("  - NEW model (50k): 15-25+ (with improvements!)")
    print("=" * 60)

if __name__ == "__main__":
    timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    quick_test(timesteps)
