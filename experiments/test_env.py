try:
    from pettingzoo.mpe import simple_spread_v3
    env = simple_spread_v3.parallel_env()
    env.reset()
    print('✓ MPE environment works!')
    print(f'Number of agents: {len(env.agents)}')
    print(f'Agents: {env.agents}')
    print(f'Observation space: {env.observation_space(env.agents[0])}')
    print(f'Action space: {env.action_space(env.agents[0])}')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()