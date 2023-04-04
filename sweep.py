import wandb
import gc
from train_network import *
from utils.train_utils import *

def sweep():
    with wandb.init():
        config = wandb.config

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PyTorch running on {device}")

        # Instantiate DataLoaders
        data_loaders = load_data(config["data_dir"], num_folds=config["num_folds"], batch_size=config["batch_size"])
        data_loaders = data_loaders[0]

        # Instantiate Networks
        model = load_model(config, device)

        # Define optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        # Train model
        train(config, device, model, data_loaders, criterion, optimizer, 
              print_every=config["verbosity"])

        # Release data back to cuda
        del model
        gc.collect()
        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    sweep_config = {
        "name": "test_sweep",
        "method": "random",
        "metric": {
            "name": "validation_loss",
            "goal": "minimize"
        },
        "parameters": {
            "use_wandb": {
                "value": True
            },
            "data_dir": {
                "value": "C:\\Users\\theow\\Documents\\Courses\\Year 4\\Thesis\\Code\\simulation_data_3v"
            },
            "num_folds": {
                "value": 5
            },
            "verbosity": {
                "value": 10
            },
            "dropout": {
                "value": 0.2
            },
            "model": {
                "value": "attention"
            },
            "epochs": {
                "value": 200
            },
            "use_checkpoint": {
                "value": False
            },
            "use_early_stop": {
                "value": True
            },
            "early_stop_patience": {
                "value": 5
            },
            "state_size": {
                "value": 3
            },
            "num_classes": {
                "value": 5
            },
            "learning_rate": {
                "values": [0.001]
            },
            "batch_size": {
                "values": [32]
            },
            "model_size": {
                "values": [16, 32, 64]
            },
            "value_size": {
                "values": [16, 32, 64]
            },
            "num_heads": {
                "min": 1,
                "max": 5
            },
            "feedforward_size": {
                "values": [16, 32, 64]
            },
            "num_encoders": {
                "min": 1,
                "max": 5
            }
        }
    }

    sweep_count = 20
    project_name = "fault_classification_cav"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=sweep, count=sweep_count)