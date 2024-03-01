1) Generate Preprocessed Data
```
python3 generate_data.py \
  --data_path raw_data.csv \
  --data_name category_classificaton_v1
```
2) Train Model
```
python3 train.py \
  --model_path base_model \
  --name model_v1 \
  --data_path category_classificaton_v1.csv
```
3) Test Model
```
python3 test.py \
  --test_path /your/test_data/path \
  --model_directory /your/model_folder/path
```
