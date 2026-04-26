import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import world_generator

app = Flask(__name__)
CORS(app)  # Allows the HTML file to communicate with this server

@app.route('/generate', methods=['GET'])
def generate():
    seed_str = request.args.get('seed')
    try:
        seed = int(seed_str)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid seed'}), 400

    print(f"Server: Generating world with seed {seed}...")
    
    # Call the existing generation logic from world_generator.py
    # world_generator.py expects int seed and handles its own file structure
    try:
        df = world_generator.generate_world(seed)
        # generate_atlas_map saves atlas.png in map_layers/{seed}/
        world_generator.generate_atlas_map(df, seed, world_generator.EXPANDED_BIOME_COLORS.copy(), save_image=True)

        # Paths to generated files (created by world_generator.py)
        output_dir = os.path.join("map_layers", str(seed))
        csv_path = os.path.join(output_dir, f"world_seed_{seed}.csv")
        atlas_path = os.path.join(output_dir, "atlas.png")

        if not os.path.exists(csv_path) or not os.path.exists(atlas_path):
            return jsonify({'error': 'Generated files not found'}), 500

        # Read and return the data
        with open(csv_path, 'r') as f:
            csv_content = f.read()

        with open(atlas_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
            img_base64 = f"data:image/png;base64,{img_data}"

        return jsonify({
            'seed': seed,
            'csv': csv_content,
            'image': img_base64
        })
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Orchestral Agent Server running on http://localhost:5000")
    app.run(port=5000)
