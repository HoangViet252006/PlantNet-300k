<p align="center">
 <h1 align="center">PlantNet-300K</h1>
</p>


## Download the dataset

- In order to train a model on the PlantNet-300K dataset, you first have to [download the dataset on Zenodo](https://zenodo.org/record/5645731#.Yuehg3ZBxPY).
- Please make sure your folder is in the location as shown below
```
├── model
├── plantnet_300K
| ├── images
│ ├── plantnet300K_metadata.json
│ └── plantnet300K_species_id_2_name.json
├── tensorboard
│ └── logs
├── inference.py
├── test.py
└── train.py

```

## How to Train
- run `python train.py --batch_size 256`
- you can follow the results while training the model by running `tensorboard --logdir tensorboard`

## How to Test
- run `python test.py --batch_size 256`
  
## How to Inference
- run `python inference.py --image_path <your image path>`
