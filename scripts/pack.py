import os
import geopandas as gpd
import pandas as pd

# 这个数据是用来进行数据打包的

class MSVDataProcessor:
    def __init__(self, data_dir):
        """
        Initialize MSVDataProcessor with a data directory
        Args:
            data_dir (str): Path to the data directory
        """
        self.msv_data = None
        self.data_dir = data_dir

    def load_data(self, file_path=None):
        """
        Load MSV data from nested directory structure
        The structure is:
        data_dir/
            large_grid_id/
                small_grid_id/
                    feature1.geojson
                    feature2.geojson
                    ...
        """
        try:
            
            self.msv_data = {}
            
            # Iterate through first level directories (large grid)
            for large_grid in os.listdir(self.data_dir):
                large_grid_path = os.path.join(self.data_dir, large_grid)
                if not os.path.isdir(large_grid_path):
                    continue
                    
                self.msv_data[large_grid] = {}
                
                # Iterate through second level directories (small grid)
                for small_grid in os.listdir(large_grid_path):
                    small_grid_path = os.path.join(large_grid_path, small_grid)
                    if not os.path.isdir(small_grid_path):
                        continue
                        
                    self.msv_data[large_grid][small_grid] = {}
                    
                    # Load all geojson files in the small grid directory
                    for geojson_file in os.listdir(small_grid_path):
                        if geojson_file.endswith('.geojson'):
                            feature_name = os.path.splitext(geojson_file)[0]
                            file_path = os.path.join(small_grid_path, geojson_file)
                            self.msv_data[large_grid][small_grid][feature_name] = gpd.read_file(file_path)
                            
        except Exception as e:
            print(f"Error loading MSV data: {e}")

    def process_data(self):
        """
        Process MSV data by combining features of the same type within each large grid
        and save the combined features to parent directory
        Returns a dictionary with large_grid_id as keys and combined features as values
        """
        if self.msv_data is None:
            raise ValueError("No data loaded")

        processed_data = {}
        parent_dir = os.path.dirname(self.data_dir)

        # Iterate through large grids
        for large_grid, small_grids in self.msv_data.items():
            feature_collections = {}
            
            # Iterate through small grids and collect features by type
            for small_grid, features in small_grids.items():
                for feature_name, gdf in features.items():
                    if feature_name not in feature_collections:
                        feature_collections[feature_name] = []
                    feature_collections[feature_name].append(gdf)
            
            # Combine features of the same type and save
            processed_data[large_grid] = {}
            for feature_name, gdfs in feature_collections.items():
                # Set CRS for each GeoDataFrame before concatenation
                gdfs = [gdf.set_crs('EPSG:4326',allow_override=True) for gdf in gdfs]
                combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
                processed_data[large_grid][feature_name] = combined_gdf
                
                # Save the combined file to parent directory
                output_file = os.path.join(parent_dir, f"{large_grid}_{feature_name}.geojson")
                combined_gdf.to_file(output_file, driver='GeoJSON')

        return processed_data

    def save_data(self, output_path):
        """
        Save processed MSV data
        """
        try:
            # TODO: Implement data saving logic
            pass
        except Exception as e:
            print(f"Error saving MSV data: {e}")


class MLDataProcessor:
    def __init__(self):
        self.ml_data = None
        self.model = None

    def load_data(self, file_path):
        """
        Load ML data from file
        """
        try:
            # TODO: Implement actual data loading logic
            pass
        except Exception as e:
            print(f"Error loading ML data: {e}")

    def preprocess_data(self):
        """
        Preprocess ML data
        """
        if self.ml_data is None:
            raise ValueError("No data loaded")
        # TODO: Implement data preprocessing logic
        pass

    def train_model(self):
        """
        Train ML model
        """
        if self.ml_data is None:
            raise ValueError("No data loaded")
        # TODO: Implement model training logic
        pass

    def predict(self, input_data):
        """
        Make predictions using trained model
        """
        if self.model is None:
            raise ValueError("Model not trained")
        # TODO: Implement prediction logic
        pass

if __name__ == "__main__":
    data_dir = "/home/gyx/projects/car2ml/data/W30/msv_all_data/data"
    msv_processor = MSVDataProcessor(data_dir)
    msv_processor.load_data()
    msv_processor.process_data()
    print("ok")