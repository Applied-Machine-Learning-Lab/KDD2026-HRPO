from models._rl_cache_model import RLCachedModelBase


class IQL(RLCachedModelBase):
    def __init__(self, args, data):
        super().__init__(args=args, data=data, model_name="IQL")

