import argparse
import cv2
import numpy as np
import os

from obelix import OBELIX

# mapping
move_choice = ["L45", "L22", "FW", "R22", "R45"]
user_input_choice = [ord("q"), ord("a"), ord("w"), ord("d"), ord("e")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--wall_obstacles", action="store_true", default=True)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--box_speed", type=int, default=2)
    args = parser.parse_args()

    bot = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )

    # 🔥 DATA STORAGE
    data = []

    # 🔥 RESET ENV
    sensor_feedback = bot.reset()
    bot.render_frame()

    episode_reward = 0

    print("\nControls:")
    print("q = L45, a = L22, w = FW, d = R22, e = R45")
    print("ESC = quit and save\n")

    for step in range(1, args.max_steps + 1):

        key = cv2.waitKey(0)

        # ESC to quit early
        if key == 27:
            print("Exiting early...")
            break

        if key in user_input_choice:

            action_idx = user_input_choice.index(key)
            action_str = move_choice[action_idx]

            #  SAVE (state, action)
            data.append((sensor_feedback.copy(), action_idx))

            # STEP ENV
            sensor_feedback, reward, done = bot.step(action_str)

            episode_reward += reward

            print(f"Step {step} | Action={action_str} | Reward={reward:.1f} | Total={episode_reward:.1f}")

            bot.render_frame()

            if done:
                print("Episode done. Total score:", episode_reward)
                break

    # ------------------------------------------------
    #  SAFE APPEND SAVE (NO OVERWRITE)
    # ------------------------------------------------
    save_path = "expert_data.npy"
    new_data = np.array(data, dtype=object)

    if len(new_data) < 50:
        print("Too few samples, not saving.")
    else:
        if os.path.exists(save_path):
            old_data = np.load(save_path, allow_pickle=True)
            combined = np.concatenate([old_data, new_data])
        else:
            combined = new_data

        np.save(save_path, combined)

        print(f"\nSaved {len(new_data)} new samples")
        print(f"Total dataset size: {len(combined)}")

    print("Done!")