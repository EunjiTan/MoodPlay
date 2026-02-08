import os
import sys
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.pipeline.orchestrator_v2 import staged_pipeline

def find_bedroom_video():
    possible_paths = [
        "uploads/bedroom.mp4",
        "bedroom.mp4",
        "uploads/Bedroom.mp4"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def main():
    print("Starting Simulation Run...")
    
    input_video = find_bedroom_video()
    if not input_video:
        print("Error: bedroom.mp4 not found!")
        return

    print(f"Using video: {input_video}")
    
    # Simulation Config
    seeds = [42, 100, 999]
    style = "cinematic"
    # Create result dir if not exists
    os.makedirs("results/simulation", exist_ok=True)

    for i, seed in enumerate(seeds):
        print(f"\n========================================")
        print(f"RUN {i+1}/{len(seeds)} | Seed: {seed}")
        print(f"========================================")
        
        # We use a unique job_id for each run to keep workspaces separate/clean
        job_id = f"sim_run_{seed}"
        output_filename = f"simulation/bedroom_seed_{seed}.mp4"
        
        staged_pipeline.run(
            input_video=input_video,
            output_name=output_filename,
            style_name=style,
            keyframe_interval=10, # Keep it reasonable for speed vs coherence
            job_id=job_id,
            clean_start=True,
            seed=seed,
            max_frames=60, # Limit to 2 seconds for faster simulation
            segment_frames=False,
            track_motion=False,
            num_inference_steps=15 # FAST Mode for CPU
        )
        
        print(f"Run {i+1} complete. Output: results/{output_filename}")

    print("\nSimulation Complete!")
    print("Check 'results/simulation/' for outputs.")

if __name__ == "__main__":
    main()
