```
./anomaly_inject_Monash/
|-- README.md
|-- dataset
|   |-- Monash
|   `-- Monash+
|-- gen_Monash+.py
|-- gen_Monash.sh
`-- injection
    |-- __init__.py
    |-- injection.py
    `-- transforms.py
```
---
- Prepare the Monash dataset.  
Monash can be downloaded at this link: https://drive.google.com/file/d/1r2l2lO713HMAzeJUJtBPOHqiV26WwX4U/view?usp=sharing  
- Running the following command to generate Monash+.
```bash
sh gen_Monash.sh
# normal time series samples' path: "./dataset/Monash+/Norm/{seq_len}_{stride}"  
# abnormal time series samples' path: "./dataset/Monash+/Anorm/{seq_len}_{stride}"
```
```python
# example
import h5py
data_path = "./dataset/Monash+/Anorm/100_50/australian_electricity_demand_dataset(0)_downsample_2.h5"
with h5py.File(data_path, 'r') as input_hf:
    data = input_hf['data'] # [num_samples, 2, 100, 1]
    series = data[:, 0]
    label = data[:, 1]
    print(series.shape) # [num_samples, 100, 1]
    print(label.shape) # [num_samples, 100, 1]
``` 
