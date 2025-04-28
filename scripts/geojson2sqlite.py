import geopandas as gpd
import os

def geojson_to_sqlite(geojson_path):
    # 读取 GeoJSON 文件
    gdf = gpd.read_file(geojson_path)
    
    # 生成同名的 sqlite 文件路径
    sqlite_path = os.path.splitext(geojson_path)[0] + '.sqlite'
    
    # 将 GeoDataFrame 写入 SQLite 数据库
    # 使用 'sqlite' 引擎，表名设为 'data'
    gdf.to_file(sqlite_path, driver='SQLite', layer='data')
    
    print(f"Converted {geojson_path} to {sqlite_path}")

if __name__ == '__main__':
    # 示例用法
    # geojson_file = "/home/gyx/projects/serinf/results/map_learn_l35725sample/test_all_vec.geojson"
    # geojson_file = "/home/gyx/projects/shapeformer/Dataset/raw/serinf2jag/modisf/20250304_2125_l35_809/train_all_vec.geojson"
    geojson_file = "/home/gyx/projects/shapeformer/Dataset/raw/serinf2jag/modisf/20250304_1052_l35_727/train_all_vec.geojson"
    geojson_to_sqlite(geojson_file)