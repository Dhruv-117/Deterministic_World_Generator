import os
import sys
import base64
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import tempfile

# Add the parent directory to sys.path to import world_generator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import world_generator

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        seed_str = query.get('seed', [None])[0]
        
        if not seed_str:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Seed is required"}')
            return

        try:
            seed = int(seed_str)
        except (ValueError, TypeError):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid seed"}')
            return

        # Use /tmp on Vercel
        with tempfile.TemporaryDirectory() as tmp_dir:
            orig_cwd = os.getcwd()
            os.chdir(tmp_dir)
            
            try:
                # Call world_generator
                df = world_generator.generate_world(seed)
                world_generator.generate_atlas_map(df, seed, world_generator.EXPANDED_BIOME_COLORS.copy(), save_image=True)

                # Paths are relative to tmp_dir
                output_dir = os.path.join("map_layers", str(seed))
                csv_path = os.path.join(output_dir, f"world_seed_{seed}.csv")
                atlas_path = os.path.join(output_dir, "atlas.png")

                if not os.path.exists(csv_path) or not os.path.exists(atlas_path):
                    raise Exception("Generated files not found in /tmp")

                with open(csv_path, 'r') as f:
                    csv_content = f.read()

                with open(atlas_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    img_base64 = f"data:image/png;base64,{img_data}"

                response_data = {
                    'seed': seed,
                    'csv': csv_content,
                    'image': img_base64
                }

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            finally:
                os.chdir(orig_cwd)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
