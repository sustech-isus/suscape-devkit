
|## install


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

susc = SuscapeDataset("dataset_root_path")

scenes = susc.get_scene_names()

print(scenes)


```