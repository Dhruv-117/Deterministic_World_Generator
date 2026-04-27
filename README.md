# 🌍 Deterministic World Generator

A seed-based procedural fantasy world generator with a real-time 3D voxel renderer.  
Enter a seed → get a fully simulated planet with tectonics, rivers, lakes, biomes, and climate — rendered in your browser as an explorable 3D voxel landscape.

---

## 🔗 Links

| | URL |
|---|---|
| 🤗 Hugging Face API | `<!-- PASTE YOUR HF SPACE URL HERE e.g. https://huggingface.co/spaces/Dhruv-117/world-generator-api -->` |
| 🌐 Vercel Frontend | `<!-- PASTE YOUR VERCEL URL HERE e.g. https://your-app.vercel.app -->` |

---

## 📁 Project Structure

```
/
├── world_generator.py      # Core world simulation engine (THE MAIN COMPONENT)
├── pipeline.py             # Offline batch layer exporter
├── server.py               # Flask API server (local + HF deployment)
├── index.html              # Frontend: Map Generator + Voxel Renderer
├── requirements.txt        # Python dependencies
├── Dockerfile              # For Hugging Face Spaces deployment
└── voxel-frontend/
    └── index.html          # Deployed to Vercel
```

---

## 🧠 How It Works (Big Picture)

```
User enters seed
      ↓
Flask server (server.py)
      ↓
world_generator.py runs 17-step simulation
      ↓
Returns: world CSV + atlas PNG (base64)
      ↓
Browser stores in IndexedDB
      ↓
User clicks "Build World" → Three.js renders voxel terrain
```

---

## 🗺️ `world_generator.py` — The Core Engine

This is the heart of the entire project. Everything else — the server, the frontend, the voxel renderer — exists purely to run this file and display its output.

It generates a **316×316 tile world** (99,856 tiles total) entirely from a single integer seed. The same seed will always produce byte-for-byte identical output, no matter when or where you run it. This is called **deterministic generation**.

### Why 316×316?

The grid divides into 8×8 tile regions (25×25 regions = 625 regions total). This lets biomes and resources be assigned at the region level for coherence, while still giving per-tile detail for rivers, elevation, and climate.

---

### The 17-Step Pipeline

Every time `generate_world(seed)` is called, it runs these steps in order. Each step feeds into the next — elevation affects temperature, temperature affects moisture, moisture determines biomes, biomes reshape elevation, elevation re-routes rivers. It's a deeply interdependent simulation.

---

#### Step 1 — Tectonic Plates

```
[01/17] Generating tectonic plates...
```

The world starts with **2–4 tectonic plates** generated using noise-distorted Voronoi tessellation. Each plate is assigned a type:

- `continental` — high ground, the backbone of landmasses
- `continental_small` — smaller elevated regions
- `oceanic` — lower ground that becomes ocean

Plates are also given movement vectors (direction + speed) which determine where their boundaries are **convergent** (plates colliding → mountains) or **divergent** (plates pulling apart → rifts/valleys).

The noise distortion means plate boundaries are never straight lines — they meander realistically like real tectonic boundaries.

---

#### Step 2 — Elevation

```
[02/17] Generating elevation...
```

Elevation is built in three layers stacked together:

**Macro layer** — broad continental shape. Uses Simplex + Perlin noise at large scale (72-tile wavelength, 5 octaves). Continental plates get a +0.3 elevation bonus, oceanic plates stay low.

**Meso layer** — tectonic chains and ridge systems. Convergent plate boundaries get an uplift bonus — tiles on collision zones are pushed up to form mountain ranges. This is where major mountain ranges come from.

**Micro layer** — fine-grained local terrain texture. Small-scale noise adds surface roughness so no area is perfectly flat.

All three layers are blended and normalized to 0.0–1.0. The result is a raw heightmap that looks geologically plausible even before any biome assignment.

---

#### Step 3 — Land/Ocean Split

```
[03/17] Enforcing 50% land / 50% ocean...
```

The raw elevation is thresholded to enforce exactly **50% land, 50% ocean**. The threshold value is found by bisection. Tiles above the threshold become land (`is_land = True`), below become ocean.

---

#### Step 4 — Ocean Connectivity

```
[04/17] Validating ocean connectivity (no inland oceans)...
  - Converted 3938 inland ocean tiles to land
```

A flood-fill check ensures all ocean tiles are connected to the world edge. Any "inland sea" pockets that got cut off are converted to land. This prevents landlocked oceans which would look wrong and break river routing.

---

#### Step 5 — Mountains

```
[05/17] Identifying mountains (15% of land)...
  - Mountain tiles: 8079 (15.00% of land)
```

The top 15% of land tiles by elevation are tagged as `is_mountain = True`. Mountains are later subdivided into sub-types (rocky mountains, snow mountains, forest mountains, glaciers) based on temperature and moisture. The 15% target is a hard enforced percentage — not a soft guideline.

---

#### Step 6 — Ocean Distance

```
[06/17] Computing distance from ocean...
  - Max distance from ocean: 96.0 tiles
```

Every land tile gets a `dist_ocean` value — how many tiles away it is from the nearest ocean tile. This is used later for:
- Temperature calculation (coastal tiles are moderated)
- Moisture calculation (distance from moisture source)
- Biome assignment (continental interiors are drier)
- River routing (rivers flow toward coast)

---

#### Step 7 — River Network

```
[07/17] Generating realistic river network...
```

This is the most complex step and the most computationally expensive. It uses a full **D8 flow direction algorithm** — the same method used in real GIS software for watershed analysis.

**How it works:**

1. **D8 Flow Directions** — For every tile, water flows to whichever of its 8 neighbours is steepest downhill. This creates a complete flow direction map for the entire continent.

2. **Basin Detection** — Tiles that have no downhill neighbour (local minima, flat areas) form "basins" — places where water would pool and get stuck. These must be resolved.

3. **Basin Carving** — Each basin is carved by lowering terrain along the least-cost path to the nearest lower elevation tile or ocean, until drainage is established. Up to 200 iterations, up to 600 tiles per basin.

4. **Flow Accumulation** — Water accumulates as it flows downstream. Each tile's flow accumulation = sum of all upstream tiles draining through it. High accumulation = major river.

5. **Snowmelt Boost** — Cold mountain tiles (temperature < 0.35) get a 2.5× flow multiplier. High-moisture tiles become spring sources (+20 flow bonus). This makes mountain rivers significantly stronger.

6. **River Tracing** — Rivers are traced from headwaters (mountain/highland tiles with high flow) downstream to the ocean. The system ensures rivers don't stop inland.

7. **River Hierarchy** — Rivers are classified by flow accumulation:
   - `stream` — flow < 30
   - `river` — flow 30–60
   - `major river` — flow > 60

8. **Meander Injection** — Flat coastal rivers get deterministic meanders injected to prevent perfectly straight river mouths.

9. **Stability Check** — After Step 16 reshapes terrain, rivers are recalculated. If the new network differs too much from the original (tile count, ocean reach, and system count are all checked against tolerances), the reshape is retried with reduced strength. This retry loop is why some seeds take much longer.

**Output:** 768–2741 river tiles, 18–91 river systems, hierarchically classified.

---

#### Step 8 — Lakes

```
[08/17] Generating lakes (targeting 2-5% of land)...
  - Lake tiles: 1406 (2.61% of land)
  - Number of lakes: 44
```

Lakes are placed as flood-filled depressions. Constraints:
- Cannot be within 8 tiles of the coastline
- Minimum size: 4 tiles (no single-tile puddles)
- Maximum size: 80 tiles
- Minimum spacing between lake centers: 14 tiles
- Desert regions get at most 0–2 lakes (weighted 70%/20%/10%)
- River-connected lakes are slightly shrunk to avoid flooding river systems

---

#### Step 9 — Distance Fields

```
[09/17] Computing distance fields for rivers and lakes...
```

Every tile gets:
- `dist_river` — distance to nearest river tile
- `dist_lake` — distance to nearest lake tile

These distance fields drive the moderating effect water has on local climate. Tiles close to rivers and lakes are more moist and slightly cooler.

---

#### Step 10 — Slope & Roughness

```
[10/17] Computing derived terrain maps...
```

- **Slope** — rate of elevation change between neighbours. High slope = steep terrain.
- **Roughness** — local elevation variance. High roughness = jagged/rocky terrain.
- **Coast distance** — refined distance used for temperature moderation.

Stored in the CSV and used by the voxel renderer for visual variation.

---

#### Step 11 — Temperature

```
[11/17] Generating natural temperature simulation...
```

Temperature per tile:

```
base_temp  = latitude_factor   (equator = hot, poles = cold)
temp      -= elevation × 0.4   (altitude cooling)
temp      -= river cooling      (exp(-dist_river / 10) × 0.05)
temp      -= lake cooling       (exp(-dist_lake / 15) × 0.08)
```

Result: realistic gradient — hot tropics, cold poles, cold mountains, moderated coastlines.

---

#### Step 12 — Moisture

```
[12/17] Generating natural moisture simulation (with wind & rain shadow)...
```

Moisture simulation:
- **Ocean moisture** — exponential decay with distance: `exp(-dist_ocean / 30)`
- **Lake moisture** — `exp(-dist_lake / 18)`
- **Wind direction** — a prevailing wind vector is generated per seed. Wind carries moisture inland.
- **Rain shadow** — tiles on the leeward (downwind) side of mountains get significantly less moisture. This is why deserts often form behind mountain ranges. Up to 80% of land can be rain-shadow affected.

---

#### Step 13 — River Climate Effects

```
[13/17] Applying river climate effects...
```

Rivers boost local moisture:
- Distance 0–1 tiles: +20% moisture
- Distance 2–3 tiles: +7.5% moisture
- Distance 4–6 tiles: +2.6% moisture
- Distance 7–8 tiles: +0.6% moisture

Major rivers get an extra +5% up to 8 tiles away. This is why river valleys are greener than surrounding terrain.

---

#### Step 14 — Islands

```
[14/17] Generating islands...
  - Island tiles: 1018 (2.47% of ocean)
```

Small island chains scattered in the ocean using seeded random placement and controlled flood-fill growth. Island biomes are assigned independently based on latitude and random moisture.

---

#### Step 15 — Biome Assignment

```
[15/17] Assigning provisional biomes with expanded classification system...
  - Expanded biome types used: 29
```

Each land tile is assigned one of **30 biomes** based on temperature, moisture, elevation, and mountain status. Assignment uses a priority-ordered rule system:

**Mountains** (`is_mountain = True`):
| Condition | Biome |
|---|---|
| Very cold, any moisture | `glacier` |
| Cold, any moisture | `snow_mountains` |
| Moderate temp, high moisture | `forest_mountains` |
| Moderate temp, low moisture | `rocky_mountains` |
| Warm, high moisture | `alpine_meadows` |

**Hills** (elevated, not mountain):
`snow_hills`, `forest_hills`, `grassy_hills`, `rocky_hills`

**Plains** (low elevation):
`grassland`, `meadow`, `steppe`, `savanna`

**Forests** (moderate-high moisture, moderate temperature):
`temperate_forest`, `woodland`, `tropical_forest`, `rainforest`

**Deserts** (very low moisture):
`sand_desert`, `rock_desert`, `badlands`, `oasis`

**Snow/Cold** (very low temperature):
`snow_plains`, `snow_forest`

**Wetlands** (very high moisture, low elevation):
`swamp`, `marsh`, `mangrove`

**Water:** `deep_ocean`, `inland_lake`, `river`, `beach`

A **seed-based variance system** means 10% of seeds get a strongly dominant biome category — some worlds are desert-heavy, others are forested, others are frozen.

---

#### Step 16 — Elevation Reshaping

```
[16/17] Reshaping elevation by biome and recomputing hydro-climate...
```

After biomes are assigned, elevation is **reshaped to match biome expectations**. Each biome has a target profile:

| Biome | Target Level | Relief Multiplier |
|---|---|---|
| `snow_mountains` | 0.98 | 1.28× |
| `rocky_mountains` | 0.96 | 1.30× |
| `glacier` | 0.99 | 0.90× |
| `grassy_hills` | 0.62 | 0.92× |
| `grassland` | 0.48 | 0.68× |
| `beach` | 0.37 | 0.44× |
| `swamp` | 0.38 | 0.46× |
| `mangrove` | 0.36 | 0.42× |

Mountains are pushed higher, wetlands are pushed lower, plains are flattened. This makes terrain visually match its biome.

After reshaping, **the entire hydrology pipeline reruns** (Steps 7–14) because changing elevation changes river routes. A stability check compares the new river network to the original. If rivers have changed too much, the reshape retries with reduced strength (60% of previous attempt). **This retry loop is the main bottleneck** — it's why Hugging Face free tier takes 8–12 minutes and why some seeds are faster than others.

---

#### Step 17 — Atlas Map

Renders a colour-coded PNG overview map using each biome's hex colour. This is the atlas image shown in the history panel and used by the voxel renderer to colour each tile.

---

### Output Files

Saved to `map_layers/{seed}/`:

| File | Contents |
|---|---|
| `world_seed_{seed}.csv` | All 99,856 tiles with 40+ data columns |
| `atlas.png` | Top-down colour map of the world |

**CSV columns include:** `x`, `y`, `elevation`, `is_land`, `is_mountain`, `is_lake`, `biome`, `temperature`, `moisture`, `dist_ocean`, `dist_river`, `dist_lake`, `slope`, `roughness`, `river_id`, `river_width`, `soil_fertility`, `iron_potential`, `copper_potential`, `gold_potential`, `coal_potential`, `fossil_fuel_potential`, `wood_biomass_capacity`, `plant_diversity_capacity`, and more.

---

### Key Constants You Can Tune

| Constant | Default | Effect |
|---|---|---|
| `WORLD_WIDTH / WORLD_HEIGHT` | 316 | Map size in tiles |
| `LAND_PERCENT` | 0.50 | Land/ocean ratio |
| `MOUNTAIN_PERCENT_OF_LAND` | 0.15 | How much terrain is mountains |
| `LAKE_PERCENT_MIN / MAX` | 0.02–0.05 | Lake coverage range |
| `TARGET_MAX_RIVERS` | 28 | Max number of river systems |
| `ELEVATION_RESHAPE_BASE_STRENGTH` | 0.54 | How aggressively biomes reshape terrain |
| `MOISTURE_OCEAN_DECAY` | 30.0 | How far inland ocean moisture reaches |

---

### Noise System

`world_generator.py` includes its own `SeededNoiseGenerator` class that wraps the `noise` library (Perlin + Simplex). If `noise` isn't available (e.g. no `gcc` to compile it), it falls back to a pure-NumPy FBM (Fractal Brownian Motion) implementation. All noise is offset by seeded random values so every seed produces a completely unique noise landscape.

---

## 🗂️ `pipeline.py` — Offline Layer Exporter

A standalone batch tool — not used by the web app. Exports the world into separate PNG image layers for use in game engines or external tools.

**Usage:**
```bash
python pipeline.py 12345
```

Exports 5 PNG layers to `map_layers/{seed}/`:

| Layer | R Channel | G Channel | B Channel |
|-------|-----------|-----------|-----------|
| `layer1_terrain.png` | Elevation | Is Mountain | Is Land |
| `layer2_water.png` | Ocean distance | River distance | Is Lake |
| `layer3_climate.png` | Temperature | Moisture | Sunlight |
| `layer4_biome.png` | Biome ID | Wood biomass | Plant diversity |
| `layer5_resources.png` | Metal ores | Soil fertility | Fossil fuels |

Each channel is normalized to 0–255. Useful for importing into Unity, Godot, or any tool that reads PNG height/data maps.

---

## ⚙️ `server.py` — Flask API

Bridges the frontend and world generator.

**Endpoint:**
```
GET /?seed=12345
```

**Response:**
```json
{
  "seed": 12345,
  "csv": "x,y,elevation,biome,...",
  "image": "data:image/png;base64,..."
}
```

**Running locally:**
```bash
pip install -r requirements.txt
python server.py
# Runs on http://127.0.0.1:5000
```

---

## 🖥️ `index.html` — The Frontend

A single-file web app with two tabs:

**Tab 1: Map Generator**
- Enter a seed (or leave blank for random)
- Calls the API (local server or HF Space depending on hostname)
- Saves result to browser **IndexedDB** (persistent local history — survives page refresh)
- Shows last 12 generated worlds as thumbnails
- Download any world as a ZIP (CSV + atlas PNG)

**Tab 2: Voxel World Renderer**
- Upload a CSV + atlas PNG manually, OR select a world from history
- Click **Build World** → Three.js renders real-time 3D voxel terrain
- Each tile becomes a coloured voxel cube. Forested biomes get trees on top.
- Two camera modes:
  - **Orbit** — left-drag to rotate, right-drag to pan, scroll to zoom
  - **Fly** — WASD move, mouse look, Q/E for up/down, Shift to sprint

**Tech stack:** Three.js r128 (3D), JSZip (downloads), IndexedDB (history), vanilla JS.

---

## 🚀 Deployment

### Option A — Hugging Face Spaces + Vercel

> ⚠️ **Performance Warning:** The free HF CPU tier is very slow for this project. World generation typically takes **8–12 minutes** on free hardware due to the river stability retry loop in Step 16. The same seed runs in ~15–30 seconds locally. If you need faster generation, use ngrok (Option B) or upgrade to a paid HF Space.

**Backend on Hugging Face:**

1. Create a Space at [huggingface.co](https://huggingface.co) → SDK: **Docker**

2. Push these files to the HF Space repo:
```
Dockerfile
server.py
world_generator.py
requirements.txt
```

3. `Dockerfile` — the `gcc` line is required; the `noise` library won't compile without it:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "server.py"]
```

4. `server.py` must bind to all interfaces on port 7860:
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
```

5. Push using your HF access token (huggingface.co/settings/tokens → Write):
```bash
git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@huggingface.co/spaces/YOUR_USERNAME/world-generator-api
git push
```

**Frontend on Vercel:**

1. Update the endpoint in `index.html` (~line 302):
```js
const endpoint = isLocal 
  ? `http://127.0.0.1:5000/generate?seed=${seed}` 
  : `https://YOUR_USERNAME-world-generator-api.hf.space/?seed=${seed}`;
```

2. Add `vercel.json` inside `voxel-frontend/`:
```json
{
  "rewrites": [{ "source": "/", "destination": "/index.html" }]
}
```

3. Push to GitHub, import on [vercel.com](https://vercel.com):
   - Root Directory: `voxel-frontend`
   - Framework: **Other**
   - Build command: empty
   - Output directory: empty

**HF free tier notes:**
- Sleeps after 48 hours of inactivity (30s–2min wake time)
- Generation takes **8–12 minutes** on free CPU
- Paid CPU upgrade ($9/mo) brings this down to ~1–2 minutes

---

### Option B — ngrok (runs on your machine, instant)

Your machine does all the work. No cloud limits, no wait times.

**Step 1 — Install ngrok:**
```bash
winget install ngrok
```

**Step 2 — Authenticate:**
```bash
ngrok config add-authtoken YOUR_TOKEN
```
Get token from [dashboard.ngrok.com](https://dashboard.ngrok.com)

**Step 3 — Run:**
```bash
# Terminal 1
python server.py

# Terminal 2
ngrok http 5000
```

ngrok gives you:
```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:5000
```

**Step 4 — Update `index.html`:**
```js
const endpoint = isLocal 
  ? `http://127.0.0.1:5000/generate?seed=${seed}` 
  : `https://abc123.ngrok-free.app/generate?seed=${seed}`;
```

Push to Vercel. Done.

**Limitations:**
- Laptop must stay on and `server.py` must be running
- Free ngrok URLs change every restart — update `index.html` each time
- Paid ngrok ($10/mo) gives a fixed permanent URL

---

## 🛠️ Local Development

```bash
pip install -r requirements.txt
python server.py
# Open index.html in browser — auto-detects localhost and calls local server
```

Generate a world without the browser:
```bash
python pipeline.py 12345
# Outputs to map_layers/12345/
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web server |
| `flask-cors` | Cross-origin requests from browser |
| `numpy` | Array math (core of all simulation) |
| `pandas` | CSV/dataframe handling |
| `scipy` | Voronoi tessellation, Gaussian filters |
| `Pillow` | PNG image generation (atlas map) |
| `opencv-python-headless` | Image processing |
| `noise` | Perlin/Simplex noise (requires `gcc` to compile) |
| `matplotlib` | Atlas colour map rendering |

---

## ⚡ Performance

| Environment | Generation Time |
|---|---|
| Local machine (modern CPU) | ~15–30 seconds |
| ngrok (routes to local machine) | ~15–30 seconds |
| **Hugging Face free CPU** | **8–12 minutes** |
| Hugging Face paid CPU upgrade | ~1–2 minutes |

Generation time varies by seed. Seeds that trigger multiple river-stability retries in Step 16 take the longest. On HF free tier, a retrying seed can exceed 15 minutes.
