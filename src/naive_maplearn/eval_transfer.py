import geopandas as gpd
from src.naive_maplearn.data_packer import MapLearnSLamPackageSet
from datetime import datetime
import pandas as pd
from enum import Enum

class ObjectType(Enum):
    OBJECT_NO_USE = 0  # 不用
    ROAD_SURFACE_LINE = 1  # 路面的标线
    ROAD_SURFACE_ARROW = 2  # 路面的箭头 
    ROAD_SURFACE_MARK = 3  # 路上的标志
    ROAD_SIGN = 4  # 路上的标牌
    ROAD_TRAFFIC_LIGHT = 5  # 路上的信号灯
    ROAD_BOUNDARY = 6  # 路边的条带
    ROAD_POLE = 7  # 路边的杆子
    ROAD_OVERHEAD = 10  # 路顶的高架设施
    VISUAL_DRIVABLE_BOUNDARY = 11  # 视觉可行驶边界
    ROAD_POINT = 12  # 路面的特征点
    OBJECT_UNKNOWN = 1000  # 未知

def eval_transfer_set(slam_meta_data:dict,\
                      mlsp: MapLearnSLamPackageSet,\
                      time_span_info:dict,\
                        slam_in_all_runs:dict,\
                            eval_result_path: str,\
                            eval_geofile_path: str):
    slam_data_gdf = generate_sqlite(slam_in_all_runs)
    eval_result = search_eval_result(eval_result_path)
    eval_gdf = get_eval_gdf(eval_geofile_path)
    
    ____count = 0
    ____all = len(mlsp.main_dict)
    with open('maplearn_eval.csv', 'w') as f:
        for mpl_id,slam_set in mlsp.main_dict.items():
            print(f"进度{____count}/{____all}\n")
            # print(f'当前地图学习数据名称{mpl_id}')
            m_et = EvalTransfer()
            m_et.load_features(mpl_id,slam_set['map_learn_type'],slam_set['isi'], slam_meta_data,slam_data_gdf)
            
            # print(m_et.vector_features)
            # 通过时间范围过滤
            m_et.filter_with_time_span(time_span_info)  
            m_et.filter_with_type()

            # 搜索对应的评价结果
            m_et.connect_eval_result(eval_result,eval_gdf)
            m_et.transfer_attribute()
            print(f"地图学习要素id:{mpl_id}\n地图学习要素关联SLAM要素:{m_et.eval_results}\n量化评价结果:{m_et.eval_val}\n",end='\r')
            
            # Save results to CSV with dictionary data
            if ____count == 0:  # Write header 
                f.write('mpl_id,eval_score,aggre_count,eval_details\n')
            
            # Count total key-value pairs in the second level of the dictionary
            total_pairs = sum(len(values) for values in m_et.eval_results.values())
            print(f"Total key-value pairs in second level: {total_pairs}")
            # Check if all values in eval_results dictionary are empty
            all_empty = all(len(values) == 0 for values in m_et.eval_results.values())
            eval_details = str(m_et.eval_results).replace(',', ';')  # Replace commas to avoid CSV confusion
            if all_empty:
                f.write(f'{mpl_id},{-1},{total_pairs},"{eval_details}"\n')
            else:
                f.write(f'{mpl_id},{m_et.eval_val},{total_pairs},"{eval_details}"\n')
            ____count+=1

def get_eval_gdf(eval_result_path:str = '/home/gyx/projects/serinf/results/map_learn_l35727/wx.geojson'):
    try:
        eval_gdf = gpd.read_file(eval_result_path)
        print(f"Successfully loaded evaluation results from {eval_result_path}")
    except FileNotFoundError:
        print(f"Evaluation result file not found at {eval_result_path}")
        eval_gdf = None
    except Exception as e:
        print(f"Error loading evaluation results: {str(e)}")
        eval_gdf = None
    return eval_gdf

def search_eval_result(eval_result_path:str='/home/gyx/projects/serinf/results/map_learn_l35727/prediction_with_subtype_original.csv'):
    """Read evaluation results from a CSV file and store them in a DataFrame
        
    Args:
        eval_result_path (str): Path to the CSV file containing evaluation results
    """
    try:
        eval_results_df = pd.read_csv(eval_result_path)
        print(f"Successfully loaded evaluation results from {eval_result_path}")
    except FileNotFoundError:
        print(f"Evaluation result file not found at {eval_result_path}")
        eval_results_df = None
    except Exception as e:
        print(f"Error loading evaluation results: {str(e)}")
        eval_results_df = None
    return eval_results_df

def generate_sqlite(slam_in_all_runs):
    # Initialize empty lists to store features
    features = []
    run_ids = []
    
    slam_data_gdf = {}


    # Iterate through each run and its features
    for run_id, features_dict in slam_in_all_runs.items():
        # Convert each feature in the run to a list
        features = []
        for oid, feature in features_dict.items():
            features.append(feature)
            run_ids.append(run_id)
    
    # Create GeoDataFrame from the features
        gdf = gpd.GeoDataFrame.from_features(features, crs='EPSG:4326')
        slam_data_gdf[run_id] = gdf
    # Add run_id column
        # gdf['run_id'] = run_ids
        gdf.to_file('dataframe_run'+str(run_id)+'.sqlite', driver='SQLite', layer='test')
        del gdf
    return slam_data_gdf
    # return gdf

class EvalTransfer:
    def __init__(self):
        """Initialize the feature transfer handler"""
        self.map_features = None  # Type A features (learned map features)
        self.vector_features = None  # Type B features (vector features)
    
    def load_features(self, map_features, map_features_type,vector_features,slam_meta_data,slam_data_gdf):
        """Load both types of features for processing
        
        Args:
            map_features: Collection of Type A features (learned map features)
            vector_features: Collection of Type B features (vector features)
        """
        self.map_features = map_features
        self.map_features_type = map_features_type
        self.vector_features = vector_features
        self.meta = slam_meta_data
        self.slam_data_gdf = slam_data_gdf
        # print(map_features,vector_features)

    def filter_with_time_span(self, time_span_info):
        """Filter vector features based on time span information
        
        Args:
            time_span_info (dict): Dictionary containing start and end time information
        """
        if not self.vector_features:
            return
        if not time_span_info:
            return
            
        start_time = time_span_info.get('start_time')
        end_time = time_span_info.get('end_time')
        
        if not start_time or not end_time:
            return
            
        filtered_features = {}
        for run_id, _features in self.vector_features.items():
            
            _gdf = self.slam_data_gdf[run_id]
            filtered_features[run_id] = []
            for _oid in _features:
                _feat_start_time = _gdf.loc[_gdf['oid'] == str(_oid)]['start_time'].values[0]
                _feat_end_time = _gdf.loc[_gdf['oid'] == str(_oid)]['end_time'].values[0]
                
                # Convert start_time and end_time to datetime objects
                start_time_dt = datetime.strptime(start_time, '%Y-%m-%d %H-%M-%S')
                end_time_dt = datetime.strptime(end_time, '%Y-%m-%d %H-%M-%S')
                
                # Convert _feat_start_time and _feat_end_time to datetime objects
                _feat_start_time_dt = datetime.fromtimestamp(_feat_start_time / 1e6)
                _feat_end_time_dt = datetime.fromtimestamp(_feat_end_time / 1e6)
                
                if _feat_start_time_dt >= start_time_dt and _feat_end_time_dt <= end_time_dt:
                    filtered_features[run_id].append(_oid)
            # print(len(_features) - len(filtered_features[run_id]))
        self.vector_features = filtered_features

    def filter_with_type(self):
        def type_trans(slam_type:str):
            if slam_type == ObjectType.ROAD_SURFACE_LINE.value:
                return 'Line'
            elif slam_type == ObjectType.ROAD_SURFACE_ARROW.value:
                return 'Arrow'
            elif slam_type == ObjectType.ROAD_SURFACE_MARK.value:
                return 'Arrow'
            elif slam_type == ObjectType.ROAD_SURFACE_LINE.value:
                return 'Boundary'
            
        if not self.vector_features:
            return
        filtered_features = {}
        for run_id, _features in self.vector_features.items():
            _gdf = self.slam_data_gdf[run_id]
            filtered_features[run_id] = []
            for _oid in _features:
                _feat_type = _gdf.loc[_gdf['oid'] == str(_oid)]['type'].values[0]
                _expected_mpl_type = type_trans(_feat_type)
                if _expected_mpl_type == self.map_features_type:
                    filtered_features[run_id].append(_oid)
            # print(len(_features) - len(filtered_features[run_id]))
        self.vector_features = filtered_features

    def connect_eval_result(self,eval_result:pd.DataFrame,eval_gdf:gpd.GeoDataFrame):
        """Connect evaluation results to vector features based on oid"""
        def get_slam_feature(run_id, slam_gdf:gpd.GeoDataFrame, \
                     eval_gdf:gpd.GeoDataFrame, \
                    oid,
                    eval_result:pd.DataFrame):
            # Get unique oids from candidate evaluation result
            # oids = _candiate_eval_result['oid'].unique()
            
            # Find corresponding features in eval_gdf where oid matches
            matching_eval_features = eval_gdf[eval_gdf['oid']==str(oid)].copy()
            original_slam_feature = slam_gdf[slam_gdf['oid']==str(oid)].iloc[0].geometry
            
            if matching_eval_features.empty:
                return None
            
            # Convert to a local projected CRS (EPSG:3857 - Web Mercator)
            matching_eval_features_proj = matching_eval_features.to_crs('EPSG:32648')
            original_slam_feature_proj = gpd.GeoSeries([original_slam_feature], crs='EPSG:4326').to_crs('EPSG:32648')[0]
            
            matching_eval_features['distance'] = matching_eval_features_proj.geometry.distance(original_slam_feature_proj)
            # Get the feature with minimum distance
            matched_feature_id = matching_eval_features.loc[matching_eval_features['distance'].idxmin()]['id']
            return matched_feature_id

        if not self.vector_features:
            return
        
        eval_results = {}
        
        for run_id, oids in self.vector_features.items():
            eval_results[run_id] = {}
            eval_result_path = f'eval_results_run{run_id}.csv'
            
            try:
                for oid in oids:
                    _candiate_eval_result = eval_result[eval_result['oid'] == oid]
                    if not _candiate_eval_result.empty:
                        if len(_candiate_eval_result) >= 2:
                            target_eval_id = get_slam_feature(run_id, self.slam_data_gdf[run_id], eval_gdf, oid, eval_result)
                        else:
                            target_eval_id = int(_candiate_eval_result['id'].iloc[0])
                        
                        flag = _candiate_eval_result[_candiate_eval_result['id'] == target_eval_id]['预测'].values[0]
                        if flag =='准确':
                            eval_results[run_id][oid] = 1
                        else:
                            eval_results[run_id][oid] = 0
            except Exception as e:
                print(f"Error processing evaluation results for run_id {run_id}: {e}")
                eval_results[run_id] = {oid: None for oid in oids}
        
        self.eval_results = eval_results

    def transfer_attribute(self):
        """Calculate the average score from eval_results.
        
        Returns:
            float: Average score (sum of valid values / count of valid entries)
            Returns 0 if no valid entries found.
        """
        eval_result = self.eval_results
        total_count = 0  # a
        total_sum = 0    # b
        
        for run_id in eval_result:
            for oid, value in eval_result[run_id].items():
                if value in (0, 1):  # Only count valid binary values
                    total_count += 1
                    total_sum += value
        self.eval_val = total_sum / total_count if total_count > 0 else 0
        return self.eval_val
    
    def _find_related_features(self, map_feature):
        """Find vector features that correspond to given map feature
        To be implemented based on specific relationship criteria
        """
        raise NotImplementedError
        
    def _aggregate_attribute(self, features, attribute_name):
        """Aggregate attribute values from multiple features
        To be implemented based on specific aggregation rules
        """
        raise NotImplementedError