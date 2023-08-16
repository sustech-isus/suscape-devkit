
## About

This is a python devkit package for the suscape dataset. It provides interfaces to load the dataset and to access the data.

## Install
Install from pypi:

```bash
pip install suscape-devkit
```

for development use:

```bash
pip install -e .
```


## verify installation

```python


from suscape.dataset import SuscapeDataset
dataset = SuscapeDataset("dataset_root_path")
print(len(dataset.get_scene_names()), 'scenes')
print(dataset.get_scene_info("scene-000000"))



```