# EDM_Custom

This is the implementation of Controlled Diffusion Network (CODNet).

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/KaziFahmi/EDM_Custom.git
```

### 2. Get a Wandb API Key
Go to ```wandb.ai/site``` and sign up. Upon logging into your account, you should get access to the API key for your account.

### 3. Train the EDM
Open the ```sbdd-diffusion.ipynb``` file.
In the first cell:
1. Insert your Wandb API key into the 'wandb_key' variable.
2. Replace <insert file address here> with the appropriate file address.
3. Ensure that the save_model variable is set to True to save the model weights.
4. Run all the cells to train the EDM. The weights of the trained model should appear as the training progresses.

### 4. Train the Controlled EDM
1. Open the sbdd-controlled-diffusion.ipynb file.
2. In the first cell:
     - Insert your Wandb API key into the wandb_key variable.
     - Specify the path to the pretrained EDM weights in the pretrained_FDD_path variable.
3. Run all the cells to train the Controlled EDM.

## Heads Up
The process described above may not function smoothly if the .ipynb files are not run in Kaggle's environment.
