import json
import os
import pickle

from sklearn import svm
from sklearn.model_selection import GridSearchCV

from utils.train_utils import load_data
from utils.save_utils import save_metrics_history

def train_and_eval(config, clf, data, print_acc=True):
    train_history = dict()
    val_history = dict()

    train_loader, val_loader = data
    # Training
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.numpy()
        inputs = inputs.reshape(inputs.shape[0], -1)
        labels = labels.numpy()

        clf.fit(inputs, labels)
        train_history["accuracy"] = clf.score(inputs, labels)

    # Validation
    for data in val_loader:
        inputs, labels = data
        inputs = inputs.numpy()
        inputs = inputs.reshape(inputs.shape[0], -1)
        labels = labels.numpy()

        val_history["accuracy"] = clf.score(inputs, labels)

    if print_acc:
        print("-------------------------------") 
        print(f'Training accuracy: {train_history["accuracy"]:.3f}')
        print(f'Validation accuracy: {val_history["accuracy"]:.3f}')

    return train_history, val_history, clf

def main():
    with open("config_svm.json", 'r') as f:
        config = json.load(f)

    # Instantiate dataloaders
    data_loaders = load_data(config["train_data_dir"], batch_size=None, num_folds=config["num_folds"])
    train_history_list = list()
    val_history_list = list()

    for fold, (train_loader, val_loader) in enumerate(data_loaders):
        print("\nFold {}:".format(fold))

        # Instantiate model
        if config["grid_search"]:
            clf = GridSearchCV(svm.SVC(), config["param_grid"])
        else:
            clf = svm.SVC(C=config["C"], gamma=config["gamma"])

        train_history, val_history, clf = train_and_eval(config, clf, (train_loader, val_loader), print_acc=config["print_accuracy"])
        
        train_history['fold'] = fold
        val_history['fold'] = fold
        train_history_list.append(train_history)
        val_history_list.append(val_history)

        with open(os.path.join(config["model_dir"], f"{config['model']}_{fold}" + ".pickle"), "wb") as f:
            pickle.dump(clf, f)

    if config["save_results"]:
        save_metrics_history(train_history_list, save_path=os.path.join(config["save_dir"], f"{config['model']}_train_history"), is_svm=True) 
        save_metrics_history(val_history_list, save_path=os.path.join(config["save_dir"], f"{config['model']}_val_history"), is_svm=True) 
        

if __name__ == "__main__":
    main()
