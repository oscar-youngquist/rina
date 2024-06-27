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

# function used to do whatever training/eval you want....
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
    
    # return the metrics that you want Raytune to track
    return results_dict


# Define the search space
# round 1
# search_space = {"learning_rate": tune.loguniform(5e-6, 1e-2),
#                 "alpha": tune.uniform(0.01, 0.25),
#                 "gamma": tune.choice([10, 20]),
#                 "SN":    tune.choice([1, 2, 4, 6])}

# # round 2 fixing SN amd gamma
# search_space = {"learning_rate": tune.loguniform(5e-6, 5e-3),
#                 "alpha": tune.uniform(0.01, 0.20),
#                 "phi_first_out": tune.choice([30, 40, 50, 60, 70, 80]),
#                 "phi_second_out": tune.choice([40, 50, 60, 70, 80, 90]),
#                 "discrim_hidden": tune.choice([10, 20, 30, 40, 50]),
#                 "dim_a": tune.choice([3,4,5,6])}

# round 3 doing everything a large number of times
search_space = {"learning_rate": tune.loguniform(5e-6, 5e-3),
                "alpha": tune.uniform(0.01, 0.20),
                "phi_first_out": tune.choice([70, 80, 90, 100, 128]),
                "phi_second_out": tune.choice([70, 80, 90, 100, 128]),
                "discrim_hidden": tune.choice([20, 30, 40, 50, 60, 80, 100, 128]),
                "dim_a":  tune.choice([6,8,10,12,14,16]),
                "gamma":  tune.choice([10, 20]),
                "SN":     tune.choice([2,4,6,8])}

# create the search algorithm we will use
algo = OptunaSearch()
# limit the number of threads that can be run in parallel
# algo = ConcurrencyLimiter(algo, max_concurrent=4)

def tune_RINA(num_samples, prefix, config_dict):
    
    options = {}
    
    # fill in the default values...
    options["dim_a"]            = 3
    options["features"]         = ['q', 'q_dot', 'tau_cmd']
    options["label"]            = 'tau_residual_cmd'
    options["labels"]           = ["FR_hip", "FR_knee", "FR_foot", "FL_hip", "FL_knee", "FL_foot",
                                   "RR_hip", "RR_knee", "RR_foot", "RL_hip", "RL_knee", "RL_foot"]
    options["dataset"]          = 'rina'
    options["learning_rate"]    = 5e-4
    options["model_save_freq"]  = 100
    options['num_epochs']       = 500
    options["gamma"]            = 10    # max 2-norm of a
    options['alpha']            = 0.01  # adversarial regularization loss weight
    options['frequency_h']      = 2.    # discriminator update frequency
    options['SN']               = 4     # maximum single-layer spectral norm of phi UPDATE - upped to 4 based on initial raytune run
    options['train_path']       = '/home/oyoungquist/Research/RINA/rina/data/lcm_converted_log/06_24_2024_formal/training_data_corrected/'
    options['test_path']        = '/home/oyoungquist/Research/RINA/rina/data/lcm_converted_log/06_24_2024_formal/eval_data_corrected/'
    # options['train_path']       = '/work/pi_hzhang2_umass_edu/oyoungquist_umass_edu/RINA/rina/data/06_24_2024_formal/training_data_corrected/'
    # options['test_path']        = '/work/pi_hzhang2_umass_edu/oyoungquist_umass_edu/RINA/rina/data/06_24_2024_formal/eval_data_corrected/'
    options["body_offset"]      = 0
    options['shuffle']          = True
    options['K_shot']           = 50 # number of K-shot for least square on a
    options['phi_shot']         = 256 # batch size for training phi
    options['loss_type']        = 'crossentropy-loss'
    options['display_progress'] = False
    options['device']           = 'cuda'

    for key in config_dict.keys():

        if key in options.keys():
            options[key] = config_dict[key]
    
    # function used to set up the objective function call
    #     passed to the search algorithm
    def train_rina(config):
        # find keys we need to copy out of args
        config_keys = list(config.keys())

        for key in config_keys:
            options[key] = config[key]

        # grab the output_path
        output_path_base = build_output_path(prefix, options)
        options['output_path'] = output_path_base
        
        # call the objective function
        return objective(options)
    

    def build_output_path(prefix, options):
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y%H:%M:%S")
        date_time = re.sub('[^0-9a-zA-Z]+', '_', date_time)

        date_time += "_cmd_res_ex_{:d}_{:d}_a{:d}_h{:d}_e{:d}".format(options['phi_first_out'],options['phi_second_out'],options['dim_a'],options['discrim_hidden'],options['num_epochs'])

        cwd = "/home/oyoungquist/Research/RINA/rina/"

        output_path_base = os.path.join(cwd, "training_results", prefix, date_time)

        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)

        return output_path_base
            
    scheduler = ASHAScheduler()

    # this limits the number of resources that each config can use while training
    #     you can also configure GPU usage through this command
    trainable_with_gpu = tune.with_resources(train_rina, {"cpu":2, "gpu":0.0625})
    
    tuner = tune.Tuner(
        trainable_with_gpu,
        # train_rina,
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
    parser = argparse.ArgumentParser(description="RINA")
    parser.add_argument('--prefix', type=str, default='', help='Prefix for training results (default: '')')
    parser.add_argument('--df-prefix', type=str, default='raytune_df', help='Prefix for training results (default: '')')
    # ['body_rp','q','body_rp_dot','q_dot','fr_contact','tau_cmd']
    parser.add_argument('--features', nargs="+", type=str, 
                        default=['q', 'q_dot', 'tau_cmd'], 
                        help='Values used an input data (default: [q,q_dot,tau_cmd])')
    parser.add_argument('--label', type=str, default='tau_residual_cmd', help='Name of training lable (target)')
    parser.add_argument('--num-samples', type=int, default=250, help='Number of sample configs (default: 250)')
    
    args = parser.parse_args()
    prefix = args.prefix
    df_prefix = args.df_prefix
    num_samples = args.num_samples

    config_dict = vars(args)

    results = tune_RINA(num_samples, prefix, config_dict)
    best_result = results.get_best_result("phi_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best test-set loss: {}".format(best_result.metrics["phi_loss"]))

    # Get a dataframe for the last reported results of all of the trials



    df = results.get_dataframe()
    df.to_csv("/home/oyoungquist/Research/RINA/rina/{:s}_df_results.csv".format(df_prefix)) 