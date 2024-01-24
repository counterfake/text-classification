1) Generate Preprocessed Data
```
python3 generate_data.py --data_path raw_data.csv --data_name category_classificaton_v1
```
2) Train Model
```
python3 train_bert.py --model_path base_model --name model_v1 --data_path category_classificaton_v1.csv
```
3) Test Model