import argparse

def get_parser(): 
    run_parser = argparse.ArgumentParser(fromfile_prefix_chars='@') 
    run_parser.add_argument("--gpu", type=str, default=None)
    run_parser.add_argument("--test", type=bool, default=False)
    run_parser.add_argument("--render", action='store_true')
    run_parser.add_argument("--board_height", type=int, default=None)
    run_parser.add_argument("--board_width", type=int, default=None)    
    run_parser.add_argument("--max_episode_timesteps", type=int, default=None)
    run_parser.add_argument("--max_epochs", type=int, default=None)
    run_parser.add_argument("--max_total_timesteps", type=int, default=None)
    run_parser.add_argument("--nb_games", type=int, default=None)
    run_parser.add_argument("--updates_per_iter", type=int, default=None)
    run_parser.add_argument("--alpha", type=float, default=None)
    run_parser.add_argument("--clip", type=float, default=None)
    run_parser.add_argument("--saving_interval", type=int, default=None)
    run_parser.add_argument("--log_dir", type=str, default=None)
    run_parser.add_argument("--load_dir", type=str, default=None)
    run_parser.add_argument("--save_dir", type=str, default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--comment", type=str, default=None)

    run_parser.add_argument("--replay_buffer_length", type=int, default=None)
    run_parser.add_argument("--training_start", type=int, default=None)
    run_parser.add_argument("--batch_size", type=int, default=None)
    run_parser.add_argument("--gamma", type=float, default=None)
    run_parser.add_argument("--init_epsilon", type=float, default=None)
    run_parser.add_argument("--final_epsilon", type=float, default=None)
    run_parser.add_argument("--epochs", type=int, default=None)
    run_parser.add_argument("--target_network_update", type=int, default=None)
    run_parser.add_argument("--epsilon_decay_period", type=int, default=None)
    run_parser.add_argument("--state_rep", type=str, default=None)
    run_parser.add_argument("--load_file", type=str, default=None)
    return run_parser