import json
import utest

def save_results(results: dict, path):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    f.close()
    
def load_results(json_path='results.json'):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: {json_path} not found. Using dummy data for demonstration.")
        return utest.generate_dummy_data()

    
