from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune import Trainable

from rina_raytune_wrap import RINA_Tune
import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime
import re

def objective(options):  
    # create a few output_paths
    eval_adapt_start = 0
    eval_adapt_end   = 2500
    eval_val_start   = 2500
    eval_val_end     = 5000

    eval_test_data_images_output_path = os.path.join(options["output_path"], "eval_images", "testing")
    eval_test_data_text_output_file   = os.path.join(options["output_path"], "eval_images", "test_set_metrics.txt")
    config_output_path                = os.path.join(options["output_path"], "config.json")

    # this will train a model for some number of epochs...
    model_trainer = RINA_Tune(options)

    # save out the config path
    model_trainer.save_config(config_output_path)

    # train the model
    model_trainer.train_model()

    # save out the statistics from training
    model_trainer.save_training_data()

    # collect the test set errors
    error_befores, error_means, error_learned = model_trainer.eval_model(eval_adapt_start, eval_adapt_end,
                                                                         eval_val_start, eval_val_end,
                                                                         eval_test_data_images_output_path,
                                                                         eval_test_data_text_output_file)
    
    # tabluate the averages (and stddevs)
    before_mean = np.mean(error_befores)
    before_std  = np.std(error_befores)
    mean_mean   = np.mean(error_means)
    mean_std    = np.std(error_means)
    phi_mean    = np.mean(error_learned)
    phi_std     = np.std(error_learned)

    
    results_dict = {}
    results_dict["phi_loss"]    = phi_mean
    results_dict["phi_std"]     = phi_std

    results_dict["avg_pred_loss"]   = mean_mean
    results_dict["avg_pred_std"]    = mean_std
    
    results_dict["before_loss"] = before_mean
    results_dict["before_std"]  = before_std

    results_dict["folder"] = options["output_path"].split("/")[-1]
    
    return results_dict

search_space = {"learning_rate": tune.loguniform(5e-6, 1e-2),
                "alpha": tune.uniform(0.01, 0.25),
                "gamma": tune.choice([10, 20]),
                "SN":    tune.choice([1, 2, 4, 6])}

algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=10)

def tune_RINA(num_samples):
    
    options = {}
    
    # fill in the default values...
    options["dim_a"]           = 3
    options["features"]        = ['q', 'q_dot', 'tau_cmd']
    options["label"]           = 'tau_residual_cmd'
    options["labels"]          = ["FR_hip", "FR_knee", "FR_foot", "FL_hip", "FL_knee", "FL_foot",
                                  "RR_hip", "RR_knee", "RR_foot", "RL_hip", "RL_knee", "RL_foot"]
    options["dataset"]         = 'rina'
    options["learning_rate"]   = 5e-4
    options["model_save_freq"] = 100
    options['num_epochs']      = 500
    options["gamma"]           = 10    # max 2-norm of a
    options['alpha']           = 0.01  # adversarial regularization loss weight
    options['frequency_h']     = 2.    # discriminator update frequency
    options['SN']              = 2     # maximum single-layer spectral norm of phi
    options['train_path']      = '/home/oyoungquist/Research/RINA/rina/data/lcm_converted_log/05_17_2024_formal/training_data/'
    options['test_path']       = '/home/oyoungquist/Research/RINA/rina/data/lcm_converted_log/05_17_2024_formal/eval_data/'
    options["body_offset"]     = 0
    options['shuffle']         = True
    options['K_shot']          = 32 # number of K-shot for least square on a
    options['phi_shot']        = 256 # batch size for training phi
    options['loss_type']       = 'crossentropy-loss'
    
    def train_rina(config):
        # find keys we need to copy out of args
        config_keys = list(config.keys())

        for key in config_keys:
            options[key] = config[key]

        # grab the output_path
        output_path_base = build_output_path()
        options['output_path'] = output_path_base
        
        # call the objective function
        return objective(options)
    

    def build_output_path():
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y%H:%M:%S")
        date_time = re.sub('[^0-9a-zA-Z]+', '_', date_time)

        date_time += "_cmd_residual"

        cwd = '/home/oyoungquist/Research/RINA/rina'

        output_path_base = os.path.join(cwd, "training_results", "raytune_initial", date_time)

        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)

        return output_path_base
            
    scheduler = ASHAScheduler(max_t=1, grace_period=1, reduction_factor=2)
    trainable_with_gpu = tune.with_resources(train_rina, {"gpu": 1, "cpu":32})
    
    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            metric="phi_loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
            scheduler=scheduler
        ),
        param_space=search_space,
    )

    return tuner.fit()


if __name__ == '__main__':
    results = tune_RINA(num_samples=100)
    best_result = results.get_best_result("phi_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best test-set loss: {}".format(best_result.metrics["phi_loss"]))

    # Get a dataframe for the last reported results of all of the trials

    df = results.get_dataframe()
    df.to_csv("/home/oyoungquist/Research/RINA/rina/raytune_df.csv") 