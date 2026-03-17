import subprocess
import sys
import os
import pandas as pd
import numpy as np
from PIL import Image

def normalize(arr):
    arr = arr.astype(float)
    min_val, max_val = arr.min(), arr.max()
    if max_val == min_val:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def csv_to_layers(csv_path, atlas_path, output_dir, map_width, map_height):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.sort_values(['y', 'x']).reset_index(drop=True)
    print(f"Loaded {len(df)} tiles")

    def col_to_grid(col_name, default=0):
        if col_name in df.columns:
            return df[col_name].fillna(default).values.reshape(map_height, map_width)
        print(f"Warning: '{col_name}' not found, using zeros")
        return np.zeros((map_height, map_width))

    # ── LAYER 1: TERRAIN ──────────────────────────────────────────
    print("Generating Layer 1: Terrain...")
    layer1 = np.stack([
        normalize(col_to_grid('elevation')),
        (col_to_grid('is_mountain') * 255).astype(np.uint8),
        (col_to_grid('is_land') * 255).astype(np.uint8)
    ], axis=2)
    Image.fromarray(layer1, 'RGB').save(f"{output_dir}/layer1_terrain.png")

    # ── LAYER 2: WATER ────────────────────────────────────────────
    print("Generating Layer 2: Water...")
    layer2 = np.stack([
        normalize(col_to_grid('dist_ocean')),
        normalize(col_to_grid('dist_river')),
        (col_to_grid('is_lake') * 255).astype(np.uint8)
    ], axis=2)
    Image.fromarray(layer2, 'RGB').save(f"{output_dir}/layer2_water.png")

    # ── LAYER 3: CLIMATE ──────────────────────────────────────────
    print("Generating Layer 3: Climate...")
    layer3 = np.stack([
        normalize(col_to_grid('temperature')),
        normalize(col_to_grid('moisture')),
        normalize(col_to_grid('sunlight_exposure'))
    ], axis=2)
    Image.fromarray(layer3, 'RGB').save(f"{output_dir}/layer3_climate.png")

    # ── LAYER 4: BIOME ────────────────────────────────────────────
    print("Generating Layer 4: Biome...")
    biome_col = df['biome'].fillna('unknown') if 'biome' in df.columns else pd.Series(['unknown'] * len(df))
    biome_encoded = pd.Categorical(biome_col).codes.reshape(map_height, map_width)
    layer4 = np.stack([
        normalize(biome_encoded),
        normalize(col_to_grid('wood_biomass_capacity')),
        normalize(col_to_grid('plant_diversity_capacity'))
    ], axis=2)
    Image.fromarray(layer4, 'RGB').save(f"{output_dir}/layer4_biome.png")

    # ── LAYER 5: RESOURCES ────────────────────────────────────────
    print("Generating Layer 5: Resources...")
    layer5 = np.stack([
        normalize(
            col_to_grid('iron_potential') +
            col_to_grid('copper_potential') +
            col_to_grid('tin_potential') +
            col_to_grid('gold_potential')
        ),
        normalize(col_to_grid('soil_fertility')),
        normalize(col_to_grid('fossil_fuel_potential') + col_to_grid('coal_potential'))
    ], axis=2)
    Image.fromarray(layer5, 'RGB').save(f"{output_dir}/layer5_resources.png")

    # ── COPY ATLAS INTO SEED FOLDER ───────────────────────────────
    if os.path.exists(atlas_path):
        from shutil import copyfile
        copyfile(atlas_path, f"{output_dir}/atlas.png")
        print(f"Atlas copied to {output_dir}/atlas.png")
    else:
        print(f"Warning: Atlas image not found at {atlas_path}")

    print(f"\n✅ Done! All files saved to: {output_dir}")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


def run_pipeline(seed):
    # ── STEP 1: RUN WORLD GENERATOR ───────────────────────────────
    print(f"\n🌍 Running world generator with seed: {seed}")
    result = subprocess.run(
        [sys.executable, "world_generator.py", str(seed)],
        check=True
    )
    print("World generator finished!")

    # ── STEP 2: LOCATE OUTPUT FILES ───────────────────────────────
    output_dir = f"map_layers/{seed}"
    csv_path   = f"{output_dir}/world_seed_{seed}.csv"
    atlas_path = f"{output_dir}/world_seed_{seed}_atlas.png"

    if not os.path.exists(csv_path):
        print(f"❌ ERROR: Expected CSV not found: {csv_path}")
        sys.exit(1)

    # ── STEP 3: DETECT MAP SIZE FROM CSV ──────────────────────────
    print("Detecting map dimensions from CSV...")
    df_check = pd.read_csv(csv_path)
    map_width  = df_check['x'].max() + 1
    map_height = df_check['y'].max() + 1
    print(f"Map size: {map_width}x{map_height} tiles")

    # ── STEP 4: GENERATE LAYERS ───────────────────────────────────
    print(f"\n🗺️  Generating map layers → {output_dir}/")
    csv_to_layers(csv_path, atlas_path, output_dir, map_width, map_height)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py {seed}")
        print("Example: python pipeline.py 12345")
        sys.exit(1)

    seed = sys.argv[1]
    run_pipeline(seed)