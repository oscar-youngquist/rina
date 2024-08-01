from rina_raytune_wrap import RINA_Tune
import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime
import re
import argparse


def run_training_loop(options):

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

    # save out plots of training/testing data
    if options['save_data_plots'] == True:
        model_trainer.save_data_plots()

    # train the model
    model_trainer.train_model()

    # save out the statistics from training
    model_trainer.save_training_data()

    # collect the test set errors
    error_befores, error_means, error_learned = model_trainer.eval_model(eval_adapt_start, eval_adapt_end,
                                                                         eval_val_start, eval_val_end,
                                                                         eval_test_data_images_output_path,
                                                                         eval_test_data_text_output_file)
    
    scripted_model_name = 'model_cmd_res_cc_a{:d}_{:d}_{:d}_h{:d}_e{:d}.pt'.format(options['dim_a'],options['phi_first_out'],options['phi_second_out'],options['discrim_hidden'],options['num_epochs'])
    scripted_model_path = os.path.join(options['output_path'], scripted_model_name)
    
    model_trainer.save_scripted_model(scripted_model_path)


def build_output_path(options):
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y%H:%M:%S")
    date_time = re.sub('[^0-9a-zA-Z]+', '_', date_time)

    date_time += "_cmd_res_cc_{:d}_{:d}_a{:d}_h{:d}_e{:d}".format(options['phi_first_out'],options['phi_second_out'],options['dim_a'],options['discrim_hidden'],options['num_epochs'])

    cwd = os.getcwd()

    output_path_base = os.path.join(cwd, "training_results", options["output_prefix"], date_time)

    if not os.path.exists(output_path_base):
        os.makedirs(output_path_base)

    options['output_path'] = output_path_base





# Index(['body_pos', 'body_ori', 'body_velo', 'body_ang_velo', 'body_acc',
#        'body_ang_acc', 'q', 'q_dot', 'q_ddot_est', 'q_ddot_m', 'tau',
#        'tau_cmd', 'fr_cmd', 'contact_m', 'fr_contact', 'tau_residual_m',
#        'tau_residual_full', 'tau_residual_cmd_centered', 'body_rpy', 'body_rp',
#        'body_rp_dot', 'steps', 'tau_residual_cmd'],
#       dtype='object')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RINA")
    
    # parser.add_argument('--train-path', type=str, 
    #                     default='/home/hcr/Research/DARoSLab/DARoS-Core/lcm_converted_log/06_24_2024_formal/training_data_corrected/', 
    #                     help='Path to training data')
    # parser.add_argument('--test-path', type=str, 
    #                     default='/home/hcr/Research/DARoSLab/DARoS-Core/lcm_converted_log/06_24_2024_formal/eval_data_corrected/', 
    #                     help='Path to eval data')
    
    parser.add_argument('--train-path', type=str, 
                        default='/home/oyoungquist/Research/RINA/rina/data/lcm_converted_log/06_24_2024_formal/training_data_cs/', 
                        help='Path to training data')
    parser.add_argument('--test-path', type=str, 
                        default='/home/oyoungquist/Research/RINA/rina/data/lcm_converted_log/06_24_2024_formal/eval_data_cs/', 
                        help='Path to eval data')
    
    # parser.add_argument('--train-path', type=str, 
    #                     default='/home/oyoungquist/Research/RINA/rina/data/lcm_converted_log/06_24_2024_formal_clean/training_data_cse_3/', 
    #                     help='Path to training data')
    # parser.add_argument('--test-path', type=str, 
    #                     default='/home/oyoungquist/Research/RINA/rina/data/lcm_converted_log/06_24_2024_formal_clean/eval_data_cse_3/', 
    #                     help='Path to eval data')
    

    # parser.add_argument('--train-path', type=str, 
    #                     default='/work/pi_hzhang2_umass_edu/oyoungquist_umass_edu/RINA/rina/data/06_24_2024_formal/training_data_corrected/', 
    #                     help='Path to training data')
    # parser.add_argument('--test-path', type=str, 
    #                     default='/work/pi_hzhang2_umass_edu/oyoungquist_umass_edu/RINA/rina/data/06_24_2024_formal/eval_data_corrected/', 
    #                     help='Path to training data')

    parser.add_argument('--num-epochs', type=int, default=10000, help='Number of epochs to train (default: 10000)')
    parser.add_argument('--learning-rate', type=float, default=0.0009, help='Learning rate (default: 0.0009)')
    parser.add_argument('--model-save-freq', type=int, default=100, help='Number of epochs between model saves (default: 100)')
    parser.add_argument('--SN', type=float, default=6.0, help='Max single-layer spectural norm (default: 6.0)')
    parser.add_argument('--gamma', type=float, default=10, help='Max magnitude of a (default: 10.0)')
    parser.add_argument('--alpha', type=float, default=0.048, help='Dis. regularization weight (default: 0.048)')
    parser.add_argument('--frequency-h', type=float, default=2.0, help='Phi/Dis. update ratio (default: 2.0)')
    parser.add_argument('--phi-first-out', type=int, default=96, help='First/Third layer size (default: 96)')
    parser.add_argument('--phi-second-out', type=int, default=128, help='Second layer size (default: 128)')
    parser.add_argument('--discrim-hidden', type=int, default=64, help='Hidden layer size for discriminator (default: 64)')
    parser.add_argument('--K-shot', type=int, default=50, help='Hidden layer size for discriminator (default: 64)')
    parser.add_argument('--phi-shot', type=int, default=2048, help='Training batch-size (default: 64)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device (default: cuda:0)')

    parser.add_argument('--shuffle', action="store_true", help='Shuffle training data (default: True)')
    parser.add_argument('--no-shuffle', dest="shuffle", action="store_false")
    
    parser.add_argument('--save-data-plots', action='store_true', help='Save out plots highlighting training data (default: True)')
    parser.add_argument('--no-save-data-plots', dest='save_data_plots', action='store_false')
    
    parser.add_argument('--display-progress', action="store_true", help='Print out progress (default: True)')
    parser.add_argument('--no-display-progress', dest="display_progress", action="store_false")


    parser.set_defaults(shuffle=True)
    parser.set_defaults(save_data_plots=True)
    parser.set_defaults(display_progress=True)


    # ['body_rp','q','body_rp_dot','q_dot','fr_contact','tau_cmd']
    parser.add_argument('--features', nargs="+", type=str, 
                        default=['q', 'q_dot', 'tau_cmd'], 
                        help='Values used an input data (default: [q,q_dot,tau_cmd])')
    parser.add_argument('--label', type=str, default='tau_residual_cmd', help='Name of training lable (target)')
    parser.add_argument('--dim-a', type=int, default=16, help='Number of basis-functions')
    parser.add_argument('--output-prefix', type=str, default='', help='Prefix for output folder (default: '')')


    args = parser.parse_args()

    options = {}

    # # fill in the default values...
    options["dim_a"]           = 16
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
    options['SN']              = 4     # maximum single-layer spectral norm of phi UPDATE - upped to 4 based on initial raytune run
    options['train_path']      = '/home/oyoungquist/Research/RINA/rina/data/lcm_converted_log/05_17_2024_formal/training_data_ex/'
    options['test_path']       = '/home/oyoungquist/Research/RINA/rina/data/lcm_converted_log/05_17_2024_formal/eval_data_ex/'
    options['shuffle']         = True
    options["body_offset"]     = 0
    options['K_shot']          = 32 # number of K-shot for least square on a
    options['phi_shot']        = 256 # batch size for training phi
    options['loss_type']       = 'crossentropy-loss'
    options['phi_first_out'] = 128
    options['phi_second_out'] = 128
    options['discrim_hidden'] = 20
    options['display_progress'] = False

    # find keys we need to copy out of args
    args_dict = vars(args)
    config_keys = args_dict.keys()

    print(args_dict)
    print("\n\n")
    print(options)

    # print(config_keys)

    for key in config_keys:
        options[key] = args_dict[key]

    print(options)

    # build the output path
    build_output_path(options)

    # print(options.keys())

    # print(options["output_path"])

    # print(options['features'])

    run_training_loop(options)


# body_rp q body_rp_dot q_dot fr_contact tau_cmd



# /work/pi_hzhang2_umass_edu/oyoungquist_umass_edu/RINA/rina/data/06_24_2024_formal/training_data

# nohup python3 train_rina.py --output-prefix cmd_residual_centered_c --num-epochs 10000 --label tau_residual_cmd_centered --discrim-hidden 64 --phi-first-out 128 --phi-second-out 128 --device cuda:0 --phi-shot 2048 > bigger_hnet_fixed_arch_test.txt &

# python3 train_rina.py --output-prefix cmd_residual_centered_c/extended_state --num-epochs 15000 --label tau_residual_cmd_centered --discrim-hidden 128 --phi-first-out 64 --phi-second-out 128 --device cuda:0 --phi-shot 2048 --K-shot 256 --features body_rp q body_rp_dot q_dot tau_cmd --no-save-data-plots

# python3 train_rina.py --output-prefix cmd_residual_centered_full/extended_state/post_ray --num-epochs 10000 --label tau_residual_full --discrim-hidden 40 --phi-first-out 80 --phi-second-out 128 --device cuda:0 --phi-shot 2048 --K-shot 320 --features body_rp q body_rp_dot q_dot tau_cmd --no-save-data-plots --learning-rate 0.003988 --alpha 0.019565 --dim-a 14 --SN 6

# python3 train_rina.py --output-prefix cmd_residual_cs/extended_state --num-epochs 10000 --label tau_residual_cmd_cs --discrim-hidden 40 --phi-first-out 80 --phi-second-out 128 --device cuda:0 --phi-shot 2048 --K-shot 320 --features body_rp q body_rp_dot q_dot tau_cmd --no-save-data-plots --learning-rate 0.003988 --alpha 0.019565 --dim-a 14 --SN 6

# python3 train_rina.py --output-prefix cmd_residual_cs/extended_state/12_5_lbs --num-epochs 10000 --label tau_residual_cmd_cs --discrim-hidden 40 --phi-first-out 100 --phi-second-out 128 --device cuda:0 --phi-shot 2048 --K-shot 320 --features body_rp q body_rp_dot q_dot tau_cmd --no-save-data-plots --learning-rate 0.00138 --alpha 0.0238 --dim-a 16 --SN 6 --gamma 10

# python3 train_rina.py --output-prefix cmd_residual_cs_update/extended_state/12_5lbs --num-epochs 10000 --label tau_residual_cmd_cs --discrim-hidden 40 --phi-first-out 100 --phi-second-out 128  --device cuda:0 --phi-shot 2048 --K-shot 320 --features body_rp q body_rp_dot q_dot tau_cmd --no-save-data-plots --learning-rate 0.00138 --alpha 0.0238 --dim-a 16 --SN 6 --gamma 10

# python3 train_rina.py --output-prefix cmd_residual_cs_update/extended_state/15lbs --num-epochs 10000 --label tau_residual_cmd_cs --discrim-hidden 40 --phi-first-out 100 --phi-second-out 128  --device cuda:0 --phi-shot 2048 --K-shot 320 --features body_rpy q body_velo body_ang_velo q_dot tau_cmd --no-save-data-plots --learning-rate 0.00138 --alpha 0.0238 --dim-a 16 --SN 6 --gamma 10

# python3 train_rina.py --output-prefix cmd_residual_cse/corrected/extended_state/15lbs --num-epochs 10000 --label tau_residual_cmd_cse --discrim-hidden 40 --phi-first-out 100 --phi-second-out 128 --device cuda:0 --phi-shot 2048 --K-shot 320 --features body_rp q body_rp_dot q_dot tau_cmd --no-save-data-plots --learning-rate 0.000345 --alpha 0.0238 --dim-a 16 --SN 6 --gamma 10