import torch
from docopt import docopt
from trainer import PPOTrainer
from yaml_parser import YamlParser

from stable_baselines3.common.utils import set_random_seed

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the config file [default: ./configs/poc_memory_env.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
        --seed=S                   Set seed [default: 0]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]
    config = YamlParser(options["--config"]).get_config()

    print("Experiment:", run_id)

    # Seed
    seed = int(options["--seed"])
    if seed != 0:
        print('Seed:', seed)
        set_random_seed(seed)
        if torch.cuda.is_available():
            print('[CUDA available]')
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print('[CUDA unavailable]')  
    else:
        print('Seed not set!')
  
    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(config, run_id=run_id, device=device)
    trainer.run_training()
    trainer.close()

if __name__ == "__main__":
    main()