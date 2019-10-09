import argparse
from agents.DQN import DQN_experiment
from agents.A2C import A2C_experiment
from agents.Rainbow import Rainbow_experiment


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="GPUMonitor - GPU utilization benchmarking tool"\
                                     "\nAlexey Tusov, tusovalexey[at]gmail.com"\
                                     "\nIrina Gorodovskaya  ir.gorod[at]gmail.com",
                                     epilog="example usage:"\
                                            "python ./main.py")
    parser.add_argument("--log_dir", dest='log_dir', type=str, default="./log/",
                        help="Directory for log files")
    parser.add_argument("--env", dest='env', type=str, default="PongNoFrameskip-v4",
                        help="Name of the gym environment used to train the agents")
    parser.add_argument("--batch_size", dest='batch_size', type=int, default=32,
                        help="Batch size input to NN")
    parser.add_argument("--max_frames", dest='max_frames', type=int, default=1000000, help="Max frames")
    parser.add_argument("--model", dest='model', type=str, default="DQN", help="Reinforcement learning model to use")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    if args.model == "DQN" or args.model == "ALL":
        DQN_experiment(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "NSTEP_DQN" or args.model == "ALL":
#        #NSTEP_DQN(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "DDQN" or args.model == "ALL":
#        #DDQN(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "Dueling_DQN" or args.model == "ALL":
#        #Dueling_DQN(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "Noisy_nets_DQN" or args.model == "ALL":
#        #Noisy_nets_DQN(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "Priority_reply_DQN" or args.model == "ALL":
#        #Priority_reply_DQN(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "Categorical_DQN" or args.model == "ALL":
#        #Categorical_DQN(args.env, args.batch_size, args.max_frames, args.log_dir)
    elif args.model == "Rainbow" or args.model == "ALL":
        Rainbow_experiment(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "QuantileRegression_DQN" or args.model == "ALL":
#        #QuantileRegression_DQN(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "Quantile_Rainbow" or args.model == "ALL":
#        #Quantile_Rainbow(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "DRQN" or args.model == "ALL":
#        #DRQN(args.env, args.batch_size, args.max_frames, args.log_dir)
    elif args.model == "A2C" or args.model == "ALL":
        A2C_experiment(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "GEA" or args.model == "ALL":
#        #GEA(args.env, args.batch_size, args.max_frames, args.log_dir)
#    elif args.model == "PPO" or args.model == "ALL":
#        #PPO(args.env, args.batch_size, args.max_frames, args.log_dir)
    else:
        print("Unsupported model")


