import numpy as np
from tqdm import tqdm
import os
from parse import parse_args
from simulation.utils import fix_seeds

from simulation.avatar import Avatar
from simulation.arena import Arena
import wandb


# load model
if __name__ == '__main__':
    args = parse_args()
    # print(args)
    fix_seeds(args.seed) # set random seed
    os.environ["OPENAI_MODEL"] = args.llm_model
    os.environ["LLM_API_STYLE"] = args.llm_api_style
    os.environ["LLM_TEMPERATURE"] = str(args.llm_temperature)
    if getattr(args, "openai_api_key", ""):
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if getattr(args, "openai_api_base", ""):
        os.environ["OPENAI_API_BASE"] = args.openai_api_base

    if(args.use_wandb):
        wandb.init(
            # set the wandb project where this run will be logged
            project = "sandbox",
            name = args.simulation_name,
            group = args.dataset
        )

    arena_ = Arena(args)
    arena_.excute()

    print('Simulation finished!')
    if(args.use_wandb):
        wandb.finish()
