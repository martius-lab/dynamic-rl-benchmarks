from gym.envs.registration import register


from .envs.drawbridge import DrawbridgeEnv
from .envs.tennis_2d import Tennis2DEnv


register(
        id = "Drawbridge-v1",
        entry_point = "dyn_rl_benchmarks.envs:DrawbridgeEnv", 
        max_episode_steps = DrawbridgeEnv.max_episode_length
        )

register( id = "Tennis2D-v1",
        entry_point = "dyn_rl_benchmarks.envs:Tennis2DEnv"
        )

register(
        id = "Tennis2DDenseReward-v1",
        entry_point = "dyn_rl_benchmarks.envs:Tennis2DDenseRewardEnv"
        )
        
register(
        id = "Platforms-v1",
        entry_point = "dyn_rl_benchmarks.envs:PlatformsEnv"
        )
        
register(
        id = "PlatformsTime-v1",
        entry_point = "dyn_rl_benchmarks.envs:PlatformsTimeEnv"
        )
