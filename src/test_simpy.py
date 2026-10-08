import simpy

def node(env, name, delay):
    yield env.timeout(delay)
    print(f"{name} finished at t={env.now}")

env = simpy.Environment()
env.process(node(env, "mist", 1))
env.process(node(env, "fog", 3))
env.process(node(env, "cloud", 7))
env.run()
