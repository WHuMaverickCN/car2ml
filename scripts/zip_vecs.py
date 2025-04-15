import os
import json

# Define the input directory and output file
# input_dir = "/data/gyx/cqc_p2_cd701_raw/datasets_AA4_0326/features/2024-03-26/000052/VecJsonData"
input_dir = "/home/gyx/data/cqc_05_serinf_trainset/L35_0727/2023-07-27/LS6A2E161NA505442_L35/VecJsonData"
output_file = "./all_vec_0727.geojson"

# Initialize the GeoJSON structure
geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

# Iterate through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".geojson"):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("type") == "FeatureCollection" and "features" in data:
                geojson_data["features"].extend(data["features"])

# Write the combined GeoJSON data to the output file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(geojson_data, f, ensure_ascii=False, indent=2)

print(f"Combined GeoJSON file saved to {output_file}")