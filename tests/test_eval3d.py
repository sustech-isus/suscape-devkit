
from suscape.dataset import SuscapeDataset

from suscape.eval.detection_3d import evaluate

import json


if __name__ == '__main__':
    susc = SuscapeDataset('./example/3d_metric')
    gt_json = susc.read_all_labels()

    with open('./metric_test/3d/res.json', 'r') as f:
        dt_json = json.load(f)

    
    metric,metric_str = evaluate(dt_json, gt_json, ["Car", 'Pedestrian'])
    print(metric_str)
    print(metric)
    
    from suscape.eval.detection_3d.evaluate_3d import DetEval3D
    eval_test = DetEval3D('./example/3d_metric',["Car", "Pedestrian"])
    metrics = eval_test.eval('./metric_test/3d/res.json')
    for key in metrics.keys():
        print(key,':' ,metrics[key])