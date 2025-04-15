from dataclasses import dataclass, field
from typing import Dict, Any,List
from src.hd_data import VehicleData
from shapely.geometry import shape

class MapLearnSLamPacker:
    def __init__(self, processed_vehicle_data:VehicleData):
        self.index = processed_vehicle_data.ST_index
        self.processed_vehicle_data = processed_vehicle_data
        self.bounding_maplearn = self.processed_vehicle_data.bounding_maplearning_data
        # Add other attributes as needed
    # @staticmethod
    def generate_maplearn_slam_package(self):
        # 获取逐个图层的空间索引
        index = self.index

        # 获取逐个图层的SLAM数据
        slam_data = self.processed_vehicle_data.items_in_tile

        # 获取地图学习数据
        maplearn_data = self.bounding_maplearn

        # 遍历
        self.search_slam_for_maplearn_item(maplearn_data, slam_data, index)

        return self.maplearn_slam_package_set

    def search_slam_for_maplearn_item(self, maplearn_obj, slam_data, index_dict):
        m_MapLearnSLamPackageSet = MapLearnSLamPackageSet()

        for maplearn_type in maplearn_obj.maplearn_data:
            maplearn_key = maplearn_type['type']
            maplearn_for_the_type = maplearn_type['data']
            if 'features' in maplearn_for_the_type:
                for target_maplearn_item in maplearn_for_the_type['features']:
                    
                    target_maplearn_item_id = target_maplearn_item['properties']['id'] 
                    # print(f'当前地图学习数据名称{target_maplearn_item_id}')
                    intersecting_slam_items = {}

                    # if target_maplearn_item_id == '1_4_wm7btr97m3dptq_25101':
                    #     input()
                    for i_run, _ in slam_data.items():
                        index = index_dict[i_run]
                        target_slams = index.intersection(shape(target_maplearn_item['geometry']).bounds)
                        

                        intersecting_slam_items[i_run]=[]
                        
                        for target_slam in target_slams:
                            intersecting_slam_items[i_run].append(target_slam)
                            # print(i_run,target_slam)
                    
                    def if_intersecting_slam_items_empty(intersecting_slam_items):
                        for i_run, _ in intersecting_slam_items.items():
                            if intersecting_slam_items[i_run]:
                                return False
                        return True
                    if if_intersecting_slam_items_empty(intersecting_slam_items):
                        continue
                    else:
                        m_MapLearnSLamPackageSet.main_dict[target_maplearn_item_id] = {'map_learn_type':maplearn_key,\
                                                                                       'isi':intersecting_slam_items}
                        # m_MapLearnSLamPackageSet.main_list.append(MapLearnSLamPackage(map_learn_data_ind=target_maplearn_item_id, \
                        #                                                     matched_slam_data=intersecting_slam_items))
        self.maplearn_slam_package_set =  m_MapLearnSLamPackageSet

@dataclass
class MapLearnSLamPackageSet:
    main_dict: Dict[str, Dict] = field(default_factory=dict)

class NaiveMapLearner:
    def __init__(self, slam_package: MapLearnSLamPackageSet):
        self.slam_package = slam_package

    def generate_map_features(self):
        # Implement the logic to generate map features based on slam_package data
        features = {}
        # Example: features['example_feature'] = self.slam_package.data['example_key']
        return features