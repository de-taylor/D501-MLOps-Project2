import os, pytest, tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.utils.validation import check_is_fitted, column_or_1d
from sklearn.linear_model import LogisticRegression

# local imports
from ml.data import process_data
from ml.model import train_model, save_model

# session scoped fixtures
# return the data itself from a file
@pytest.fixture(scope="session")
def data():
    """
    Returns the train/test data splits for other functions to use.
    """
    data_path = os.path.join(os.getcwd(), "data", "census.csv")
    data = pd.read_csv(data_path)

    return data

# return the split data as a tuple (train, test)
@pytest.fixture(scope="session")
def split_data(data):
    # in order train, test
    return train_test_split(data, test_size=0.3, random_state=42, shuffle=True)

# return the cateorical features used in the training script
@pytest.fixture(scope="session")
def cat_features():
    return [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(scope="session")
def trained_components(split_data, cat_features):
    # run the test run first to grab the encoder, lb, and std_scaler
    X_train, y_train, encoder, lb, std_scaler = process_data(
        split_data[0],
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=OneHotEncoder,
        std_scaler=StandardScaler
    )
    return {
        "encoder": encoder,
        "lb": lb,
        "std_scaler": std_scaler,
        "X_train": X_train,
        "y_train": y_train
    }

# testing the data itself, basic deterministic/descriptive tests
# Test Suite:
    # Data is of correct shape for train/test split, before processing
    # Numeric data columns have acceptable means and standard deviations
    # Categorical columns only have the expected values based on source data cards

def test_data_shape(data, split_data):
    # training data should be same columns as original and ~70% of the original shape
    assert split_data[0].shape[1] == data.shape[1] # columns
    assert (split_data[0].shape[0] >= data.shape[0] * 0.69) and (split_data[0].shape[0] <= data.shape[0] * 0.71) # rows

    # testing data should be same columns as original and ~30% of the original shape
    assert split_data[1].shape[1] == data.shape[1] # columns
    assert (split_data[1].shape[0] >= data.shape[0] * 0.29) and (split_data[1].shape[0] <= data.shape[0] * 0.31) # rows

def test_cont_column_ranges(split_data):
    ranges = {
        "age": (16,110),
        "fnlgt": (0,1500000),
        "education-num": (1.0, 16.0),
        "capital-gain": (0.0, 1000000.0),
        "capital-loss": (0.0, 1000000.0),
        "hours-per-week": (1.0, 100.0)
    }
    # loop through train then test data, to verify that both conform
    for dataset in split_data:
        for col_name, (minimum, maximum) in ranges.items():
            assert dataset[col_name].dropna().between(minimum, maximum).all(), (
                f"Column {col_name} failed. Should be between {minimum} and {maximum}.",
                f"Instead, min={dataset[col_name].min()} and max={dataset[col_name].max()}"
            )

def test_cat_column_values(split_data):
    cat_values = {
        "workclass": [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
            "?"
        ],
        "education": [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool"
        ],
        "marital-status": [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse"
        ],
        "occupation": [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            'Transport-moving',
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
            "?"
        ],
        "relationship": [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried"
        ],
        "race": [
            "White",
            "Asian-Pac-Islander",
            "Amer-Indian-Eskimo",
            "Other",
            "Black"
        ],
        "sex": [
            "Female",
            "Male"
        ],
        "native-country": [
            'United-States',
            'Mexico',
            '?',
            'Philippines',
            'Germany',
            'Canada',
            'Puerto-Rico',
            'El-Salvador',
            'India',
            'Cuba',
            'England',
            'Jamaica',
            'South',
            'China',
            'Italy',
            'Dominican-Republic',
            'Vietnam',
            'Guatemala',
            'Japan',
            'Poland',
            'Columbia',
            'Taiwan',
            'Haiti',
            'Iran',
            'Portugal',
            'Nicaragua','Peru',
            'France',
            'Greece',
            'Ecuador',
            'Ireland',
            'Hong',
            'Trinadad&Tobago',
            'Cambodia',
            'Thailand',
            'Laos',
            'Yugoslavia',
            'Outlying-US(Guam-USVI-etc)',
            'Honduras',
            'Hungary',
            'Scotland',
            'Holand-Netherlands'
        ],
        "salary": [
            ">50K",
            "<=50K"
        ]
    }

    for dataset in split_data:
        for col_name, values in cat_values.items():
            assert dataset[col_name].dropna().isin(values).all() == True, (
                f"Column {col_name} failed. Should be in range {str(values)}.",
                f"Instead, values={str(dataset[col_name].value_counts().index.values)}"
            )


# Test out expectations for the ml.data.process_data() function
def test_process_data_train(split_data, cat_features):
    """
    Tests out the process_data function on the training=True pathway. Verifies
    that all outputs are expected, and are of the correct type
    """
    # run the process_data function as a Training run
    x_train, y_train, encoder, lb, std_scaler = process_data(
        split_data[0],
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=OneHotEncoder,
        std_scaler=StandardScaler
    )

    # x_train - np.ndarray, 14 columns, row shape same as split_data
    assert isinstance(x_train, np.ndarray) == True
    assert x_train.shape[1] > 14 # should now have more columns than original
    assert x_train.shape[0] == split_data[0].shape[0] # same row count

    # y_train - np.ndarray, not empty, same row count as split_data
    assert isinstance(y_train, np.ndarray) == True
    column_or_1d(y_train) # 1 entry in tuple after ravel
    assert y_train.shape[0] == split_data[0].shape[0] # will only have rows, no columns

    # encoder - OneHotEncoder, test if trained?
    assert isinstance(encoder, OneHotEncoder) == True
    # will throw an exception if not fitted
    check_is_fitted(encoder) # type: ignore

    # lb - label binarizer
    assert isinstance(lb, LabelBinarizer)
    check_is_fitted(lb)

    # std_scaler - StandardScaler
    assert isinstance(std_scaler, StandardScaler)
    check_is_fitted(std_scaler)

def test_process_data_test(split_data, cat_features, trained_components):
    """
    Tests out the process_data function on the training=false pathway. Verifies
    that all outputs are expected, and are of the correct type
    """
    # run the Test run
    X_test, y_test, _, _, _ = process_data(
        split_data[1],
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=trained_components["encoder"],
        lb=trained_components["lb"],
        std_scaler=trained_components["std_scaler"]
    )
    # x_train - np.ndarray, 14 columns, row shape same as split_data
    assert isinstance(X_test, np.ndarray) == True
    assert X_test.shape[1] > 14 # should now have more columns than original
    assert X_test.shape[0] == split_data[1].shape[0] # same row count

    # y_train - np.ndarray, not empty, same row count as split_data
    assert isinstance(y_test, np.ndarray) == True
    column_or_1d(y_test) # 1 entry in tuple after ravel
    assert y_test.shape[0] == split_data[1].shape[0] # will only have rows, no columns


# testing the train_model function
def test_train_model(trained_components):
    """
    Testing the train_model function to ensure that it creates a
    trained LogisticRegression model with the expected test
    parameters. Test parameters chosen at random.
    """
    # passing in test parameters
    parameters = {
        'C': [0.1],
        'max_iter': [250],
        'solver': ['liblinear'],
        'penalty': ['l1']
    }

    model = train_model(
        X_train=trained_components["X_train"],
        y_train=trained_components["y_train"],
        parameters=parameters
    )

    model_params = model.get_params()

    # check the model itself for type and whether is fitted
    assert isinstance(model, LogisticRegression)
    check_is_fitted(model)

    # check key parameters
    assert model_params["C"] == 0.1
    assert model_params["max_iter"] == 250
    assert model_params["solver"] == "liblinear"
    assert model_params["penalty"] == "l1"


# testing the save_model function
def test_save_model(trained_components):
    """
    Test the save_model function with the encoders from the
    trained_components
    """
    # temp test path
    with tempfile.TemporaryDirectory() as tmpdirpath:
        save_model(
            trained_components["encoder"],
            os.path.join(tmpdirpath, "enc.pkl")
        )
        save_model(
            trained_components["lb"],
            os.path.join(tmpdirpath, "lb.pkl")
        )

        # test that files exist and are binary files
        assert os.path.exists(os.path.join(tmpdirpath, "enc.pkl"))
        assert os.path.exists(os.path.join(tmpdirpath, "lb.pkl"))