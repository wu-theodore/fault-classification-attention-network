import wandb
from train_network import *
from utils.train_utils import *

def sweep():
    with wandb.init():
        config = wandb.config

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PyTorch running on {device}")

        # Instantiate DataLoaders
        train_loader, val_loader = load_data(config["data_dir"], config["train_val_split"], batch_size=config["batch_size"])

        # Instantiate Networks
        model = load_model(config, device)

        # Define optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        # Train model
        train(config, device, model, (train_loader, val_loader), criterion, optimizer, print_every=config["verbosity"], use_wandb=True)

if __name__ == "__main__":
    sweep_config = {
        "name": "test_sweep",
        "method": "random",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 4
        },
        "metric": {
            "name": "validation_loss",
            "goal": "minimize"
        },
        "parameters": {
            "data_dir": {
                "value": "C:\\Users\\theow\\Documents\\Courses\\Year 4\\Thesis\\Code\\simulation_data_4_vehicles"
            },
            "train_val_split": {
                "value": 0.80
            },
            "verbosity": {
                "value": 10
            },
            "model": {
                "value": "attention"
            },
            "epochs": {
                "value": 200
            },
            "state_size": {
                "value": 4
            },
            "num_classes": {
                "value": 5
            },
            "learning_rate": {
                "values": [0.001, 0.0003]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "model_size": {
                "values": [8, 16, 32, 64]
            },
            "value_size": {
                "values": [16, 32, 64, 128, 256]
            },
            "num_heads": {
                "min": 1,
                "max": 16
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

    sweep_count = 1
    project_name = "fault_classification_cav"
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=sweep, project=project_name, count=sweep_count)