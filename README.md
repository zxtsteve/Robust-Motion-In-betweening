# Robust-Motion-In-betweening
> PyTorch Implementation of 'Robust Motion In-betweening'


This uses [`LAFAN1`](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) dataset.

## Setup

1. Clone `LAFAN1` Dataset.
   Your directory wil look like this:
   ```
   .
  ├── README.md
  ├── main.py
  └── ubisoft-laforge-animation-dataset
   ```

2. Run `evaluate.py` to unzip and validate it.
   ```bash
   $ python ubisoft-laforge-animation-dataset/evaluate.py 
   ```

## Reference

* LAFAN1
  ```
  @article{harvey2020robust,
  author    = {Félix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal},
  title     = {Robust Motion In-Betweening},
  booktitle = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
  publisher = {ACM},
  volume    = {39},
  number    = {4},
  year      = {2020}
  }
  ```
