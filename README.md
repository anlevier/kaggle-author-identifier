# Task Description

https://uazhlt-ms-program.github.io/ling-582-fall-2024-course-blog/ashtonlevier/class-competition

`git clone git@github.com:uazhlt-ms-program/ling-582-fall-2024-class-competition-code-anlevier.git`

- Navigate to the root
- To train:
  - `python3 -m src.main train --file_path=data/train.csv`

- To test:
  - Follow "To train" instructions
  - `python3 -m src.main test --file_path=data/test.csv --model_path=trained_model.h5 --tokenizer_path=tokenizer.pkl --max_len_path=max_len.pkl`
  - View results at `data/test_predictions.csv`

- To analyze:
  - Follow "To train" instructions
  - `python3 -m train_utils.held_out --input_file=data/train.csv --output_file=data/held_out.csv`
  - `python3 -m src.main test --file_path=data/held_out.csv --model_path=trained_model.h5 --tokenizer_path=tokenizer.pkl --max_len_path=max_len.pkl`
  - `python3 -m src.main analyze --file_path=data/held_out.csv --file_path2=data/test_predictions.csv --output_file=data/comparison.csv`
