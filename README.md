# TPZSFD

## Project Structure

```text
.
├── Train.py 
├── EDI_value.py 
├── LFSI_Qwen.py
├── Model.py
├── TF_attributes_fs.py
├── Data_split.py 
├── dataset_cache/ 
│  ├── MotorB_feature_cache.pkl
│  ├── seen_full_dataset.npz
│  ├── unseen_full_dataset.npz
├── Data/ 
│  ├── class_1.csv 
│  ├── class_2.csv 
│  └── ... 
├── README.md
```

-  A zero-shot fault diagnosis dataset is provided in this repository. The corresponding feature cache file is also included to reduce the computational cost and facilitate reproduction of the reported results.
-  The time-frequency statistical feature library is defined in TF_attributes_fs.py. The only required input parameter is the sampling frequency.
-  To run the framework, simply execute Train.py. All required modules and functions will be called.
-  In addition to the required Python packages, users need to download the weights of Qwen2.5-1.5B-Instruct and specify the local path in LFSI_Qwen.py before running the feature subset inference module. The model weights can be obtained from: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
-  If LLM-based feature subset inference is not required, users may directly employ the predefined SELECTED_FEATURES in Train.py to perform zero-shot learning and evaluation, bypassing the feature optimization stage.
