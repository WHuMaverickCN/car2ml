## 评价结果传递
在`config_bounding.yaml`配置中添加如下配置：

`tiles_path`配置的是瓦片文件。

- 配置周度的多趟次矢量化数据路径
```
vector_data:
  target_path:
    - /home/gyx/data/cqc_05_serinf_trainset/L35_0727_part1
    - /home/gyx/data/cqc_05_serinf_trainset/L35_0727_part2
    - /home/gyx/data/cqc_05_serinf_trainset/L35_0727_part3
```
其中，`target_path`存放的是多趟的矢量要素文件，以列表形式读取。

- 配置车端评价结果路径

```
slam_eval_result:
  eval_result_path: /home/gyx/projects/CrowdQC/inference/test_case_c385_l35_0727/prediction_with_subtype_original.csv
  eval_geofile_path: /home/gyx/projects/CrowdQC/inference/test_case_c385_l35_0727/evaluation_results.geojson

```
slam_eval_result字段中`eval_result_path`和`eval_geofile_path`分别对应三角化评价算法输出的列表和矢量数据

- 配置待评价地图学习数据路径


```
map_learning_settings:
  # target_maplearn_data_dir: /home/gyx/projects/car2ml/data/W30/ml_l35/556168550
  # target_maplearn_data_dir: /home/gyx/projects/car2ml/data/W30/ml_l35/556168546
  target_maplearn_data_dir: /home/gyx/projects/car2ml/data/W30/ml_l35/556168503
  gt_dir: /home/gyx/data/cqc/gt/groundtruth/
  week_info: W30
  route_info: L35
```

运行如下命令，执行评价结果传递
```
conda activate si_cqc
python bounding.py
```

运行如下命令，执行评价结果精度验证
```
python valid2.py
```
输出结果为`maplearn_eval.csv`文件
其形式如下
```
mpl_id,eval_score,aggre_count,eval_details
1_4_wm7btrzj8fzw2j_27673,1.0,4,"{0: {659: 1; 26794: 1}; 1: {23699: 1; 775: 1}; 2: {}}"
1_4_wm7btrm6ybxdvx_25952,1.0,12,"{0: {27032: 1; 26958: 1; 26959: 1; 26985: 1}; 1: {23870: 1; 927: 1; 23859: 1; 23896: 1}; 2: {734: 1; 740: 1; 774: 1; 739: 1}}"
1_4_wm7btprer4ktzv_24481,0.6666666666666666,3,"{0: {27389: 0; 27353: 1}; 1: {}; 2: {1037: 1}}"
1_4_wm7btpqj13d5zz_24635,1.0,5,"{0: {27467: 1; 1170: 1}; 1: {1226: 1; 24204: 1; 1232: 1}; 2: {}}"
1_4_wm7btp81p9d98m_24419,0.75,4,"{0: {27642: 1; 27617: 1}; 1: {24368: 0}; 2: {1279: 1}}"
1_4_wm7btp6pxwy6hm_24718,1.0,4,"{0: {27584: 1; 1315: 1}; 1: {24310: 1}; 2: {1187: 1}}"
1_4_wm7btp98n7yrrb_24646,1.0,3,"{0: {1369: 1; 27592: 1}; 1: {24321: 1}; 2: {}}"
1_4_wm7btp927s2n4q_24599,1.0,5,"{0: {1347: 1; 27626: 1}; 1: {24283: 1; 24329: 1}; 2: {1248: 1}}"
1_4_wm7btpkmp5e6vf_24781,1.0,1,"{0: {}; 1: {}; 2: {24293: 1}}"
1_4_wm7btr73t65qeq_25231,0.5714285714285714,7,"{0: {27138: 1; 26353: 0; 26368: 0; 26380: 0}; 1: {23158: 1; 23940: 1; 23971: 1}; 2: {}}"
1_4_wm7btr73dgtk9r_25231,0.6666666666666666,3,"{0: {26354: 0}; 1: {23145: 1; 23979: 1}; 2: {}}"
1_4_wm7btr4x9ey1mq_25034,1.0,2,"{0: {}; 1: {46293: 1; 46279: 1}; 2: {}}"
1_4_wm7btr96g74q44_25038,0.7142857142857143,7,"{0: {26522: 1; 263: 1; 240: 0}; 1: {385: 1; 23328: 0}; 2: {253: 1; 228: 1}}"
1_4_wm7btr97h63tp8_25069,0.7,10,"{0: {26522: 1; 267: 1; 263: 1; 26501: 1; 241: 0}; 1: {385: 1; 23308: 0; 23328: 0}; 2: {253: 1; 229: 1}}"
1_4_wm7btr97m3dptq_25101,0.7,10,"{0: {26522: 1; 267: 1; 263: 1; 26502: 1; 242: 0}; 1: {385: 1; 23309: 0; 23328: 0}; 2: {253: 1; 230: 1}}"
```

- 测试结果如下

| 测试用例       | 数据量 | 别名       | 精度     |
|----------------|--------|------------|----------|
| Test_case_1    | 872    | 556168550  | 91.14%   |
| Test_case_2    | 590    | 556168503  | 94.29%   |

