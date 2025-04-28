from src.hd_data import VehicleData,MapLearningData
from src.naive_maplearn.data_packer import MapLearnSLamPacker
from src.naive_maplearn.eval_transfer import eval_transfer_set
import yaml
import os
import glob

config_path = './config_bounding.yaml'

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
if not config.get('general_config', {}).get('if_indexed', True):
    # Delete spatial index files before starting
    for file in glob.glob("spatial_index_*.*"):
        if file.endswith('.dat') or file.endswith('.idx'):
            try:
                os.remove(file)
                print(f"Deleted {file}")
            except OSError as e:
                print(f"Error deleting {file}: {e}")

# 1 加载车端数据
vehicle_data = VehicleData(config_path)
vehicle_data.get_vec_data()
vehicle_data.build_rtree()


# target_maplearn_data_dir = '/home/gyx/projects/car2ml/data/W30/ml_l35/556168550'
# gt_dir = "/home/gyx/data/cqc/gt/groundtruth/"

# Read target_maplearn_data_dir from config
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
target_maplearn_data_dir = config['map_learning_settings']['target_maplearn_data_dir']
# gt_dir = config['map_learning_settings']['gt_dir']

# 2 加载地图学习数据
maplearn_data = MapLearningData(data_dir = target_maplearn_data_dir)

# 3 绑定车端数据到地图学习数据
vehicle_data.get_target_maplearning_data(maplearn_data)

# 4 遍历SLAM数据的多个图层，获取这个地图学习数据对应的瓦片id
vehicle_data.get_items_in_tile()

# 5 生成地图学习数据和SLAM数据的对应关系 存储于maplearn_slam_package_set对象中
maplearn_slam_package_set = MapLearnSLamPacker(vehicle_data).generate_maplearn_slam_package()
slam_in_all_runs = vehicle_data.vehicle_data_oid2feature


# 6 从配置文件中读取地图学习数据的配置
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

week_info = config['map_learning_settings'].get('week_info')
route_info = config['map_learning_settings'].get('route_info')

if week_info is None or route_info is None:
    print("week_info or route_info is not configured. Skipping time_span_info calculation.")
    time_span_info = None
else:
    time_span_info = config.get('inherent_map_info', {}).get(route_info, {}).get(week_info)

# time_span_info = config.get('inherent_map_info').get(route_info).get(week_info)

# Get eval result paths from config
eval_result_path = config['slam_eval_result']['eval_result_path']
eval_geofile_path = config['slam_eval_result']['eval_geofile_path']

eval_transfer_set(vehicle_data.meta_data,\
      maplearn_slam_package_set,
      time_span_info,
      slam_in_all_runs,
      eval_result_path,
      eval_geofile_path)

