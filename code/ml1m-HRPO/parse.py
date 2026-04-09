import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # Overall settings
    parser.add_argument('--simulation_name', type=str, default= 'Test',
                        help='The name of one trial of simulation.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify which gpu to use.')
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--items_per_page', type=int, default=4,
                        help='Number of items per page.')
    parser.add_argument('--num_avatars', type=int, default=20,
                        help='Number of avatars for sandbox simulation.')
    parser.add_argument('--execution_mode', type=str, default= 'parallel',
                        choices=['serial', 'parallel'],
                        help='Specify execution mode: serial or parallel.')
    parser.add_argument('--llm_model', type=str, default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                        help='LLM model used by avatars.')
    parser.add_argument('--llm_api_style', type=str, default=os.getenv("LLM_API_STYLE", "chat_completions"),
                        choices=['chat_completions', 'responses'],
                        help='Which API style to call for LLM inference.')
    parser.add_argument(
        '--llm_temperature',
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        help='Sampling temperature used by avatar LLM calls. Use 0 for stable evaluation.',
    )
    parser.add_argument(
        "--beauty_prompt_mode",
        type=str,
        default=os.getenv("BEAUTY_PROMPT_MODE", "a"),
        choices=["a", "b", "c"],
        help="Beauty prompt mode: a=single-stage (ALIGN->WATCH->RATING), b=two-stage (SCAN then DECIDE), c=strict-intent (mission-driven, conservative clicks).",
    )
    parser.add_argument(
        "--enable_hazard_plan",
        action="store_true",
        help="Enable dynamic hazard/survival-aware reranking on top of the base recommender ranking.",
    )
    parser.add_argument(
        "--hazard_plan_dir",
        type=str,
        default=os.getenv("HAZARD_PLAN_DIR", "Saved"),
        help="Artifact subdir under recommenders/weights/<dataset>/HazardPlan/.",
    )
    parser.add_argument(
        "--hazard_plan_candidate_pool",
        type=int,
        default=50,
        help="How many base-ranked unseen items to consider per page before hazard-plan reranking.",
    )
    parser.add_argument(
        "--hazard_plan_override",
        type=str,
        default=os.getenv("HAZARD_PLAN_OVERRIDE", "auto"),
        choices=["auto", "safe_match", "recover", "explore", "balanced"],
        help="Override the dynamic high-level plan used by the hazard-plan reranker.",
    )
    parser.add_argument(
        "--tiger_attr_profile",
        type=str,
        default=os.getenv("TIGER_ATTR_PROFILE", "default"),
        choices=["default", "scope_attrv3"],
        help="Optional TIGER-specific SID decode attribution profile.",
    )
    parser.add_argument(
        "--tiger_attr_history_items",
        type=int,
        default=int(os.getenv("TIGER_ATTR_HISTORY_ITEMS", "12")),
        help="How many recent history items TIGER uses for SID attribution.",
    )
    # Optional: allow passing key/base via CLI (useful on Windows / reproducibility)
    parser.add_argument('--openai_api_key', type=str, default=os.getenv("OPENAI_API_KEY", ""),
                        help='OpenAI API key. If provided, overrides env OPENAI_API_KEY.')
    parser.add_argument('--openai_api_base', type=str, default=os.getenv("OPENAI_API_BASE", ""),
                        help='OpenAI API base URL for OpenAI-compatible providers. If provided, overrides env OPENAI_API_BASE.')

    # Only recommend ground truth
    parser.add_argument("--rec_gt", action="store_true",
                        help="whether to recommend ground truth")
    
    # Using wandb
    parser.add_argument("--use_wandb", action="store_true",
                        help="whether to use wandb")
    
    # Only validate the effectiveness of agents
    parser.add_argument("--val_users", action="store_true",
                        help="whether to validate users")
    parser.add_argument('--val_ratio', type=int, default=1,
                        help='Ratio of unobserved items vs ground truth for validation.')
    
    # Advertisement settings
    parser.add_argument("--add_advert", action="store_true",
                        help="whether to add advertisement")
    parser.add_argument("--display_advert", action="store_true",
                        help="whether to display advertisement")
    parser.add_argument('--advert_type', type=str, default='pop_high',
                        choices=['all', 'pop_high', 'pop_low', 'unpop_high', 'unpop_low'],
                        help='Specify advertisement type.')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset to use.')

    # Avatar settings
    parser.add_argument('--n_avatars', type=int, default=3,
                        help='How many avatars to simulate.')
    parser.add_argument('--max_pages', type=int, default=1,
                        help='The maximum page number users would like to view')


    # Recommender settings
    parser.add_argument('--model_path', type=str, default= 'Saved',
                        help='Specify model save path.')
    parser.add_argument('--modeltype', type=str, default= 'LightGCN',
                        help='Specify model save path.')

    # others
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument("--pred_norm", action="store_true",
                        help="pred_norm")

    args, _ = parser.parse_known_args()

    return args
