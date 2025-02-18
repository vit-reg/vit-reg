# Reproducibility Study of “Vision Transformers Need Registers”

This repository contains the code needed to reproduce the experiments from the paper *"Vision Transformers Need Registers"* ([Darcet et al., 2023](https://arxiv.org/abs/2309.16588)).

The main claims regarding register tokens were validated on two training paradigms:
1. **Supervised training** for classification.
   - DEIT-III ([Touvron et al., 2022](https://arxiv.org/abs/2204.07118))
2. **Self-supervised methods** for learning visual features using the following architectures:
   - DINO ([Caron et al., 2021](https://arxiv.org/abs/2104.14294))
   - DINOv2 ([Oquab et al., 2024](https://arxiv.org/abs/2304.07193))

## Repository Structure

The repository is organized such that each directory is named after the model being considered, which is a copy of the original repository. In each directory, one can find specific notebooks that guide through our work and experiments. 

## How to Run the Notebooks

To run the notebooks, first configure the virtual environment from the repository. To do so, once you are in the project directory, execute the following commands:

```
conda env create -f environment.yml
conda activate fact
```


The directories and their corresponding notebooks are as follows:

### DEIT-III
- `deit_notebook_final.ipynb`
- `deit_fine_tuning.ipynb` – A similar set of experiments run on a fine-tuned version of the DEIT-III small model.

### DINO
- `dino_notebook_final.ipynb`

### DINOv2
- `dinov2_notebook_final.ipynb`



---
### Note  
The models `dino_vitbase16_pretrain` and `dinov2_vitl14_pretrain.pth` need to be downloaded separately from the official DINO and DINOv2 repository.

For access to the fine-tuned versions of the DEIT-III small model, please contact the developers for the model weights.


## How to run fine-tuning
In order to perform model fine-tuning, the original DEIT-III codebase was modified. Now, it is possible to specify the number of frozen layers as well as the number of registers to introduce. An example command to run fine-tuning is provided below:

```
srun python -m torch.distributed.launch \
--nproc_per_node 2 \
--use_env DEIT-III/main.py \
--num_workers 8 \
--epochs 10 \
--batch-size 256 \
--data-set IMNET \
--data-path [DATA_PATH] \
--model deit_small_patch16_LS_reg \
--output_dir [OUTPUT_DIR] \
--freeze-layers 10 \
--num-registers 4 \
--l2-weight 0.001 \
--pretrained-21k \
--reg-use-pretrained # whether to use pretrained DEIT-III model 
```

