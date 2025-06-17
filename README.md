# HYU-AUE8088, Understanding and Utilizing Deep Learning

## Project. Multispectral Pedestrian Detection

### Important Files

```bash
├── README.md
├── requirements.txt
├── datasets
│   └── kaist-rgbt/ (see below explanation)
├── data
│   ├── ...
│   └── kaist-rgbt.yaml
├── models
│   ├── ...
│   ├── common.py
│   ├── yolo.py
│   ├── yolo5n_kaist-rgbt.yaml
│   └── yolo5s_kaist-rgbt.yaml
├── utils
│   ├── ...
│   ├── eval
│   │   ├── coco.py
│   │   ├── cocoeval.py
│   │   └── kaisteval.py
│   ├── dataloaders.py
│   └── loss.py
├── detect.py
├── debug_kaist.ipynb
├── val.py
└── train_simple.py
```

### Preparation (One time setting)

- (Recommended) Set W&B environment variables
  Plase find API Keys from https://wandb.ai/settings
  ```bash
  $ echo "export WANDB_API_KEY={YOUR_WANDB_API_KEY}" >> ~/.bashrc
  $ echo "export WANDB_ENTITY={USER_NAME}" >> ~/.bashrc
  $ echo "export WANDB_DIR={YOUR_WANDB_LOG_DIR}" >> ~/.bashrc
  $ source ~/.bashrc

  ```

  Please refer this webpage for more details: https://docs.wandb.ai/ko/guides/track/environment-variables/


- Prepare dataset (5.8GB, multispectral(visible + lwir) images with bbox labels)
  ```bash
  $ wget http://ircv-nas.hanyang.ac.kr:9088/downloads/kaist-rgbt-aue8088.tar
  $ tar xvf kaist-rgbt-aue8088.tar
  ```


- Clone base code repository (replace `ircvlab` to `YOUR_GITHUB_ACCOUNT` if you forked the repository)
  ```bash
  $ git clone https://github.com/ircvlab/aue8088
  $ cd aue8088
  ```


- Create python virtual environment
  ```bash
  $ python3 -m venv venv/aue8088
  $ source venv/aue8088/bin/activate
  ```

- Check whether the virtual environment set properly
: The result should end with `venv/aue8088/bin/python`.

  ```bash
  $ which python
  ```


- [!] Create a symbolic link for kaist-rgbt dataset
    - Assume the below folder structure

      ```bash
      ├── kaist-rgbt
      ├── aue8088
      │   ├── data/
      │   ├── models/
      │   ├── train_simple.py
      │   ├── ...
      │   └── README.md (this file)
      ```

    - Use below commands
      ```bash
      $ mkdir datasets
      $ ln -s $(realpath ../kaist-rgbt) datasets/kaist-rgbt
      $
      ```

- Install required packages
  ```bash
  $ pip install -r requirements.txt
  ```


### Train
- Command
  ```bash
  $ python train_simple.py \
    --img 640 \
    --batch-size 16 \
    --epochs 20 \
    --data data/kaist-rgbt.yaml \
    --cfg models/yolov5n_kaist-rgbt.yaml \
    --weights yolov5n.pt \
    --workers 16 \
    --name yolov5n-rgbt \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls
  ```

### Evaluation (eval.ai server)
- On your labtop, go to the website: `http://166.104.168.170:8888/` (Open: 5/23)
    - Only available in Hanyang internal network
    - If you're not in campus, please use VPN (https://vpn.hanyang.ac.kr)
        - It takes a day (or two) to get the permission from IT department.
- Sign up
- Send a message to me via LMS (then, I'll manually verify your account.)
- Go to `All Challenges` - `Multispectral Pedestrian Detection Challenge` - `Submit`
- Upload your predictions on `test-all-20.txt`
    - If you run `train_simple.py` with the default setting, predictions on `test-all-20.txt` will be generated: `runs/train/*/epoch*_predictions.json`
    - You can download this file onto your computer.
    - Note: if size of the prediction file is too large (about > 30MB), evaluation on the server could be failed.