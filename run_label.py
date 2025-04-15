from src.hd_data import *

# maplearn_data = MapLearningData(data_dir='/home/gyx/projects/car2ml/data/W30/ml_l35/556168550', \
#                                 ground_truth_dir="/home/gyx/data/cqc/gt/groundtruth/")

maplearn_data = MapLearningData(data_dir='/home/gyx/projects/car2ml/data/W30/ml_l35/556168546', \
                                ground_truth_dir="/home/gyx/data/cqc/gt/groundtruth/")
maplearn_data.label()
# print(maplearn_data.ground_truth_data)
# print("run_label.py")