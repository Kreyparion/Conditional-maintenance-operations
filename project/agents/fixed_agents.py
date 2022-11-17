from project.agents.agent import Agent
from project.env.environnement import Environnement
from project.env.states import State
from project.env.actions import Action
from project.tools.logger import logger, init_logger


class FixedAgent(Agent):
    def __init__(self, env: Environnement, corr_thrs=0.1, nerf_thrs=0.25, **kwargs):
        super().__init__(env, **kwargs)
        self.env = env
        self.corr_thrs = corr_thrs
        self.nerf_thrs = nerf_thrs

    def act(self, state: State):
        items = state.items
        properties = [
            (1 - item.wear / item.threshold, item.is_nerfed) for item in items
        ]
        properties.sort()

        limits = Action()._limitationsList

        corr_num = 0
        i0 = 0
        for i in range(int(1 / limits[1])):
            # corrective actions
            i0 = i
            if properties[i][0] != 0.0:
                break
            corr_num += 1

        nb_pre_max = int((1 - corr_num * limits[1]) / limits[2])
        pre_num = 0
        while True:
            if (
                properties[i0][0] != 0
                and properties[i0][0] < self.corr_thrs
                and pre_num < nb_pre_max
            ):
                pre_num += 1
                i0 += 1
            else:
                break

        properties.sort(key=lambda x: (x[1], x[0]))
        nb_nerf = 0
        for is_nerfed, wear in properties:
            if is_nerfed or wear > self.nerf_thrs:
                break
            nb_nerf += 1

        return Action.fromDictInt({0: nb_nerf, 1: pre_num, 2: corr_num})

    def observe(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ):
        self.current_action = action
        self.current_state = state
        self.current_reward = reward
        self.done = done
        logger.info(
            f"Step : {self.env.step_number-1} Agent observes: state={state}, action={action}, reward={reward}, next_state={next_state}, done={done}"
        )

    def learn(self):
        pass

    def random(self):
        pass

    def reset(self):
        pass
