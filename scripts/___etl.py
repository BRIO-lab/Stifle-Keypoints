import sys
import os
import click
from importlib import import_module
from _____utility import set_logger
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split as tts



@click.command()
@click.argument("config_file", type=str, default="config")
def etl(config_file):

    """
    ETL function that will load raw data and convert it into pre-defined test, val, train split

    Args: config_file [str]: path to config file

    Returns: None
    """
    
    """
    Load Config
    """

    config_dir = os.getcwd() + "/config/"

    sys.path.append(config_dir)

    config_module = import_module(config_file)

    config = config_module.Configuration()

    dir_name = config.data_constants['MODEL_NAME']

    log_dir = os.path.join("log/",dir_name)

    if os.path.isdir(log_dir) == False:
        os.mkdir(log_dir)
    
    """
    CONFIGURE LOGGER
    """

    logger = set_logger(log_dir + '/etl_' + config.data_constants['MODEL_NAME'] + '.log')

    logger.info(f'Load config from {config_file}')

    raw_data_file = config.etl['raw_data_file']
    
    processed_path = Path(config.etl['processed_path'])

    test_size = config.etl['test_size']

    val_size = config.etl['val_size']

    random_state = config.etl['random_state']

    logger.info(f'config: {config.etl}')

    """
    TEST, TRAIN, VAL split
    """

    logger.info("------------------------Test, Train, Val split------------------------")

    data = pd.read_csv(raw_data_file)

    train, test = tts(data, test_size=test_size, random_state=random_state)
    train, val = tts(train, test_size=val_size, random_state=random_state)

    """
    Make a separate directory for each model if it does not exist already
    """
    new_dir = os.path.join(config.etl['processed_path'],config.data_constants['MODEL_NAME'])
    if os.path.isdir(new_dir) == False:
        os.makedirs(new_dir)
    train_name = 'train_' + config.data_constants['MODEL_NAME'] + '.csv'
    val_name = 'val_' + config.data_constants['MODEL_NAME'] + '.csv'
    test_name = 'test_' + config.data_constants['MODEL_NAME'] + '.csv'

    # Check if the dir for the train, test, and evaluate csvs exists. If not, create it
    processed_path = Path(os.path.join(config.etl['processed_path'],config.data_constants['MODEL_NAME']))

    # Write the outputted data to a csv, in the processed/model_name dir
    logger.info(f'Write data to {processed_path}')
    train.to_csv(processed_path / train_name, index=False)
    test.to_csv(processed_path / test_name, index=False)
    val.to_csv(processed_path / val_name, index = False)

    logger.info(f'Train: {train.shape}')
    logger.info(f"Val: {val.shape}")
    logger.info(f"Test: {test.shape}")
    logger.info("\n")


if __name__ == "__main__":
    etl()
