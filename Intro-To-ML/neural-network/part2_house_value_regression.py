import itertools
import torch
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold


class Regressor(nn.Module):

    def __init__(
        self,
        x,
        nb_epoch=1000,
        batch_size=100,
        neurons_per_layers=[10],
        activation_functions=["relu"],
        learning_rate=1e-3
    ):
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """
        # Inherit attributes from nn.Module (we want to make Regressor a pytorch module)
        super().__init__()

        # Set the device
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        # Catching error if these parameters are incompatible
        if len(neurons_per_layers) != len(activation_functions):
            raise ValueError(
                "The number of neuron layers does not match the number of activation functions."
            )

        # Data attributes
        self.x = x
        X, _ = self._preprocessor(self.x, training=True)

        # Pre-processing parameters
        self.scaler_x = None
        self.scaler_y = None
        self.encoder = None
        self.ratio = None

        # Hyperparameters attributes
        self.output_size = 1
        self.input_size = X.shape[1]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activation_functions = activation_functions
        self.neurons_per_layers = neurons_per_layers
        self.criterion = nn.MSELoss() # Maybe we can change to cross entropy loss if we have time

        # Constructing Network Layers
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(self.input_size, self.neurons_per_layers[0]))

        for index, value in enumerate(self.activation_functions):
            if value == "relu":
                self.layers.append(nn.ReLU())
            if value == "sigmoid":
                self.layers.append(nn.Sigmoid())
            if value == "tanh":
                self.layers.append(nn.Tanh())

            if index != len(self.activation_functions) - 1:
                self.layers.append(
                    nn.Linear(self.neurons_per_layers[index], self.neurons_per_layers[index + 1])
                )
            else:
                self.layers.append(nn.Linear(self.neurons_per_layers[index], self.output_size))

        return

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        # Split the data into numerical and categorical
        number_x = x.iloc[:, :-1]
        category_x = x[["ocean_proximity"]]

        # Deal with null data, all the nans are in total_bedrooms, so we use sklearn impute with stategy mean
        non_null_rows = x.dropna(subset=['total_rooms', 'total_bedrooms'], axis=0)
        non_null_rows = non_null_rows[non_null_rows["total_bedrooms"] > 0]
        self.ratio = (non_null_rows["total_rooms"] / non_null_rows["total_bedrooms"]).mean()
        number_x["total_bedrooms"] = number_x.apply(
            lambda row: (
                row["total_rooms"] / self.ratio
                if pd.isnull(row["total_bedrooms"])
                else row["total_bedrooms"]
            ),
            axis=1,
        )

        # Deal with any other numerical data that is null with mean fill
        number_x.fillna(number_x.mean(), inplace=True)
        
        # Deal with categtorical data by selecting random category
        category_x.fillna(
            np.unique(category_x)[np.random.randint(0, len(np.unique(category_x)))]
        )

        # If training, we save the meta data for the scaler and encoder
        if training:

            # Standardise numerical features
            scaler_x = StandardScaler()
            scaler_x.fit(number_x)
            self.scaler_x = scaler_x
            number_x = scaler_x.transform(number_x)

            # Encode categorical data
            enc = OneHotEncoder()
            enc.fit(category_x)
            self.encoder = enc
            category_x = enc.transform(category_x).toarray()

            # Standardise output if it exists
            if y is not None:
                scaler_y = StandardScaler()
                scaler_y.fit(y)
                self.scaler_y = scaler_y
                y = scaler_y.transform(y)

        else:  # Not training
            # Standardise numerical features
            number_x = self.scaler_x.transform(number_x)

            # Standardise output if it exists
            if y is not None:
                y = self.scaler_y.transform(y)

            # Encode categorical data
            category_x = self.encoder.transform(category_x).toarray()

        # transfrom to pytorch tensor
        x = torch.tensor(np.concatenate((number_x, category_x), axis = 1), dtype=torch.float32, device=self.device)
        if y is not None:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        # Return preprocessed x and y
        return x, y

    def forward(self, x):
        """
        forward function of the network.

        Arguments:
            - x {torch.tensor} -- input data of shape
                (batch_size, input_size).

        Returns:
            self {torch.tensor} -- output of that layer.

        """
        return self.layers(x)

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw outp[put array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        # Split data into training and validation
        x, x_validation, y, y_validation = train_test_split(x, y, test_size=0.1)

        # Set up the data
        X_train, Y_train = self._preprocessor(x, y=y, training=True)
        X_val, Y_val = self._preprocessor(x_validation, y=y_validation, training=False)

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

        # Set model to device
        self = self.to(self.device)

        # Set the optimiser - adaptive learning rate using Adam optimiser
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Set early stopping parameters
        patience = 10
        counter = 0
        min_val_loss = float("inf")



        # Train the model
        print(f"Device: {self.device}")
        start = time.time()
        for epoch in range(200):

            # State model is being trained
            self.train()


            # Separate data into batches and shuffle
            batch_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )

            for (x_batch, y_batch) in batch_loader:
                # Call the forward function on each batch
                y_pred = self(x_batch)

                # Compute the loss
                loss = self.criterion(y_pred, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()


                # Update parameters
                optimizer.step()

            # Calculate validation loss
            with torch.no_grad():
                val_predictions = self(X_val)
                val_loss = self.criterion(val_predictions, Y_val)

            # Early stopping
            if val_loss < min_val_loss - 0.01:
                min_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    print(f"Early stopping at epoch {epoch}, Validation loss = {val_loss}")
                    break


            # Print the loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Iteration {epoch}, Validation loss = {val_loss}")

        end = time.time()
        print(f"Model training time: {end - start}")
        self.validation_loss = min_val_loss

        return self

    
    def fit_with_tracking(self, x_train, y_train, x_validation, y_validation, i):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw outp[put array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        

        # Set up the data
        X_train, Y_train = self._preprocessor(x_train, y=y_train, training=True)
        X_val, Y_val = self._preprocessor(x_validation, y=y_validation, training=False)

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

        # Set model to device
        self = self.to(self.device)

        # Set the optimiser - adaptive learning rate using Adam optimiser
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Set early stopping parameters
        patience = 10
        counter = 0
        min_val_loss = float("inf")


        # Create lists to store loss so we can display on chart later
        eval_data = []
        models = []

        # Train the model
        print(f"Device: {self.device}")
        start = time.time()
        for epoch in range(200):

            # State model is being trained
            self.train()

            # Create list to store batch loss
            batch_loss = []

            # Separate data into batches and shuffle
            batch_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )

            for (x_batch, y_batch) in batch_loader:
                # Call the forward function on each batch
                y_pred = self(x_batch)

                # Compute the loss
                loss = self.criterion(y_pred, y_batch)
                batch_loss.append(loss.item())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()


                # Update parameters
                optimizer.step()

            training_loss = np.mean(batch_loss)

            # Calculate validation loss
            with torch.no_grad():
                val_predictions = self(X_val)
                val_loss = self.criterion(val_predictions, Y_val)

            eval_data.append({"epoch": epoch ,"training_loss": training_loss, "validation loss": val_loss })
    
            # Early stopping
            if val_loss < min_val_loss - 0.01:
                min_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    print(f"Early stopping at epoch {epoch}, Validation loss = {val_loss}")
                    break


            # Print the loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Iteration {epoch}, Validation loss = {val_loss}")
    


            end = time.time()
            print(f"Model training time: {end - start}")
            self.validation_loss = min_val_loss
            models.append(self)
                
            
            df = pd.DataFrame(eval_data)
            df.to_csv(f"Best_model_eval{i}.csv")

        return models


    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """
        # Preprocess the input
        X, _ = self._preprocessor(x, training=False)

        # predict the output
        with torch.no_grad():
            predictions = self(X.to(self.device))
        
        return self.scaler_y.inverse_transform(predictions.numpy())

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
        # Prepare the data
        y_pred = self.predict(x)
        y = y.to_numpy(dtype=np.float32)

        # Compute the mean squared error
        rmse =  np.mean((y - y_pred) ** 2).item()**0.5
        

        return rmse

def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open("part2_model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open("part2_model.pickle", "rb") as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

def generate_combinations(parameters):
    """
    Generate all possible combinations of hyperparameters.

    Arguments:
        - parameters {dict} -- Dictionary of hyperparameters to search.

    Returns:
        - {list} -- List of all possible dictionaries combinations of hyperparameters.

    """

    # Generate the configurations
    configurations = []
    for num_layers in parameters["num_layers"]:
        for first_hl in parameters["neurons_for_first_hl"]:
            # Generate all combinations of neurons per layer and activation functions for the current number of layers
            neurons_per_layers = [int(first_hl * (0.5 ** i)) for i in range(num_layers)]
            activation_functions = list(itertools.product(parameters["activation_functions"], repeat=num_layers))

            # Generate all combinations of the other parameters
            other_parameters = list(itertools.product(parameters["learning_rate"], parameters["batch_size"]))

            # Combine the configurations
            for other_params, activations in itertools.product(other_parameters, activation_functions):
                configuration = {
                    "nb_epoch": 200,
                    "learning_rate": other_params[0],
                    "batch_size": other_params[1],
                    "neurons_per_layers": neurons_per_layers,
                    "activation_functions": activations,
                }
                configurations.append(configuration)
            
    return configurations

def generate_random_combinations(parameters, n):
    """
    Generate n random combinations of hyperparameters.

    Arguments:
        - parameters {dict} -- Dictionary of hyperparameters to search.
        - n {int} -- Number of random configurations to generate.

    Returns:
        - {list} -- List of n random dictionaries combinations of hyperparameters.

    """

    # Generate the configurations
    configurations = []
    for _ in range(n):
        num_layers = np.random.choice(parameters["num_layers"])
        first_hl = np.random.choice(parameters["neurons_for_first_hl"])
        configuration = {
            "nb_epoch": 200,
            "learning_rate": np.random.choice(parameters["learning_rate"]),
            "batch_size": int(np.random.choice(parameters["batch_size"])),
            "neurons_per_layers": [int(first_hl * (0.5 ** i)) for i in range(num_layers)],
            "activation_functions": np.random.choice(parameters["activation_functions"], size=num_layers),
        }
        configurations.append(configuration)

    return configurations


def perform_hyperparameter_search(x, y):
    """
    Performs a hyper-parameter search for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
        - y {pd.DataFraravel()  # or np.array(predictions) dependin

    Returns:
        - {Regressor} -- Best model found during the search.
        - {dict} -- Best hyper-parameters found during the search.

    """
    # Define the hyperparameters to search
    parameters = {
        "learning_rate": [0.0001, 0.001, 0.01],
        "batch_size": [16, 32, 64],
        "num_layers": [1, 2, 3],
        "neurons_for_first_hl": [8, 12, 16],
        "activation_functions": ["relu", "sigmoid", "tanh"],
    }

    top_20_models = []

    # Initialise the list for storing the results
    feat_list = []

    # Perform intial parameter search
    configurations = generate_combinations(parameters)
    for configuration in configurations:
        print(configuration)
        regressor = Regressor(x, **configuration)
        regressor.fit(x, y)
        if len(top_20_models) == 0:
            top_20_models.append({**configuration, "score": regressor.validation_loss})
        else:
            for i, model in enumerate(top_20_models):
                if regressor.validation_loss < model["score"]:
                    top_20_models.insert(i, {**configuration, "score": regressor.validation_loss})
                    if len(top_20_models) > 20:
                         top_20_models.pop()
                    break
        # Store the results
        row = {**configuration, "score": regressor.validation_loss}
        feat_list.append(row)
    
    df = pd.DataFrame(feat_list)
    df.to_csv("hyperparameter_search3.csv")
    
    # Initialise the best model and score
    best_params = None
    best_model = None
    best_score = float("inf")

    
    n_splits = 5

    # Create a new dictionary from configuration that doesn't include the "score" key
    top_20_without_score = [{k: v for k, v in configuration.items() if k != "score"} for configuration in top_20_models ]
    print(top_20_without_score)
    # Find the best model amongst top 20 models
    for configuration in top_20_without_score:
        print(configuration)
        average_validation_score = 0

        # Catching error if these parameters are incompatible
        try:
            # Split the training set into 5 folds for cross-validation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_index, validation_index in kf.split(x):
                x_train, x_validation = (
                    x.iloc[train_index],
                    x.iloc[validation_index],
                )
                y_train, y_validation = (
                    y.iloc[train_index],
                    y.iloc[validation_index],
                )

                # Train the model
                regressor = Regressor(x, **configuration)
                regressor.fit(x_train, y_train)
                average_validation_score += regressor.score(
                    x_validation, y_validation
                )

            # Update the best model and score if necessary
            average_validation_score /= n_splits
            if average_validation_score < best_score:
                best_params = configuration
                best_score = average_validation_score
                best_model = regressor



        except ValueError as e:
            print(f"Skipping configuration due to error: {e}")
            continue

    # Print the best model and score and save the results
    print(f"best score: {best_score} \n best params: {best_params}")

    return best_model, best_params

def validate_best_model(x, y, x_test, y_test, model_params):
    """
        Function to cross validate best model and print out graphs to justify choice
    """

    test_scores = []
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for i, (train_index, validation_index) in enumerate(kf.split(x)):
        x_train, x_validation = (
            x.iloc[train_index],
            x.iloc[validation_index],
        )
        y_train, y_validation = (
            y.iloc[train_index],
            y.iloc[validation_index],
        )
        model = Regressor(x, **model_params)
        model.fit_with_tracking(x_train, y_train, x_validation, y_validation, i)
        test_scores.append(model.score(x_test, y_test))
    print(np.mean(test_scores))
    print(np.std(test_scores))
    
    
    
    return


def example_main():

    # Read input data as panda CSV as LabTS use it too
    data = pd.read_csv("housing.csv")

    # oversampling island in csv
    island_instances = data[data['ocean_proximity'] == 'ISLAND']
    replication_factor = 200 // len(island_instances)
    oversampled_island_instances = pd.concat([island_instances] * replication_factor, ignore_index=True)
    balanced_data = pd.concat([data[data['ocean_proximity'] != 'ISLAND'], oversampled_island_instances], ignore_index=True)

    # Split data into feature and output
    output_label = "median_house_value"
    feature = balanced_data.loc[:, balanced_data.columns != output_label]
    output = balanced_data.loc[:, [output_label]]

    # Split data into training and testing
    feature_train, feature_test, output_train, output_test = train_test_split(feature, output, test_size=0.1)


    # # Train a model
    # regressor = Regressor(feature, batch_size=100, neurons_per_layers=[16,16,16], activation_functions=["relu", "relu", "relu"])
    # regressor.fit(feature_train, output_train)
    # print(regressor.score(feature_test, output_test))

    # # Hyperparameter Tuning
    # best_model, best_params = perform_hyperparameter_search(feature_train, output_train)

    # print(best_params)

    # save_regressor(best_model)

    # # retrain model with best params and save output
    best_params = {'nb_epoch': 1000, 'learning_rate': 0.01, 'batch_size': 32, 'neurons_per_layers': [12, 6], 'activation_functions': ['tanh', 'sigmoid']}
    validate_best_model(feature_train, output_train, feature_test, output_test, best_params)
    # best_model = Regressor(feature, **best_params)
    # best_model.fit(feature, output)
    # save_regressor(best_model)
    # configurations = generate_combinations({
    #     "learning_rate": [0.0001, 0.001, 0.01],
    #     "batch_size": [32, 64, 128],
    #     "num_layers": [1, 2, 3],
    #     "neurons_per_layers": [16, 32, 64],
    #     "activation_functions": ["relu", "sigmoid", "tanh"],
    # })

    # regressor = Regressor(feature)
    # regressor.fit(feature, output)
    # regressor.predict(feature)
    # regressor.score(feature, output)

    # model = load_regressor()
    # print(model.score(feature, output))

    # print(len(configurations))


if __name__ == "__main__":
    example_main()
