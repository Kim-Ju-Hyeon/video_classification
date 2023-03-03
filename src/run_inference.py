import click
import traceback
import os
import datetime
import pytz
from easydict import EasyDict as edict
import yaml
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import setup_logging
from utils.train_helper import set_seed, mkdir, edict2dict
from runner.runner import Runner
from datasets.video_dataset import video_dataset
from torch.utils.data import DataLoader


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
@click.option('--inference', type=bool, default=True)
def main(conf_file_path, inference):
    if inference:
        log_save_name = 'Inference_log_exp'
    else:
        log_save_name = 'Train_Resume'
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
    log_file = os.path.join(config.exp_sub_dir, f"{log_save_name}_log_exp_{config.seed}.txt")
    logger = setup_logging('INFO', log_file, logger_name=str(config.seed))
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.exp_name))
    
    test_df = pd.read_csv('./data/test.csv')
    test_df['video_path'] = test_df['video_path'].apply(lambda x: '../src/data/' + x[2:])
    
    test_dataset = video_dataset(test_df['video_path'].values, None)
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size)

    try:
        runner = Runner(config=config, logger=logger)
        
        if not inference:
            config.train_resume = True
            
            train_df = pd.read_csv('./data/train.csv')
            train_df['video_path'] = train_df['video_path'].apply(lambda x: '../src/data/' + x[2:])

            train, val, _, _ = train_test_split(train_df, train_df['label'], test_size=0.2, random_state=config.seed)

            train_dataset = video_dataset(train['video_path'].values, train['label'].values)
            val_dataset = video_dataset(val['video_path'].values, val['label'].values)
            
            train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=config.train.batch_size)
            
            runner.train(train_dataloader, val_dataloader)

        preds = runner.test(test_dataloader)
        
        submit = pd.read_csv('./data/sample_submission.csv')
        submit['label'] = preds
        submit.to_csv(os.path.join(config.exp_sub_dir, 'submit.csv'), index=False)
        
    except:
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
