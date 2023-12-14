import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from scipy.stats import entropy


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 4
        num_landmarks = 2
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.20

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # landmark.state.p_pos = np.array([-0.6+i*1.2, -0.4])  # 修改处，将动态的目标修改为了静态的目标
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #     rew = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
    #         rew -= min(dists)
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):  # 有个bug，自己和自己发生碰撞，但不影响
    #                 rew -= 1
    #     return rew

    # def observation(self, agent, world):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for entity in world.landmarks:  # world.entities:
    #         entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     # entity colors
    #     entity_color = []
    #     for entity in world.landmarks:  # world.entities:
    #         entity_color.append(entity.color)
    #     # communication of all other agents
    #     comm = []
    #     other_pos = []
    #     for other in world.agents:
    #         if other is agent: continue
    #         comm.append(other.state.c)
    #         other_pos.append(other.state.p_pos - agent.state.p_pos)
    #     return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    def observation(self, agent, world):
        """
        各个agent与各个land的距离，与其他agent的相对位置， 与各个land的相对位置，自己的位置和速度，当前分布，目标分布，总的agent数目
        """
        differ_distribution = [0, 0]
        need_transition = [0, 0]
        land = [0, 0]  # 每个land上的agent数目

        entity_pos = []  # 与各个land的相对位置
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        other_pos = []  # 与其他agent的相对位置
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        self_pos = agent.state.p_pos  # 自己的位置与速度
        self_vel = agent.state.p_vel

        agent_land_dis = []  # 每一个agent与每一个land的距离
        for a in world.agents:
            dist = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.landmarks]
            min_index = dist.index(min(dist))
            land[min_index] += 1
            agent_land_dis.append(dist)
        agent_land_alldis = np.array(agent_land_dis).flatten().tolist()

        # 目标分布（1/4， 3/4），目标分布减去当前分布
        total_sum = sum(land)
        normalized_land = [value / total_sum for value in land]

        differ_distribution[0] = (world.target_distribute[0] - normalized_land[0])  # 调整状态
        differ_distribution[1] = (world.target_distribute[1] - normalized_land[1])

        total_agent_num = np.array([total_sum])  # 所有智能体的数目
        need_transition[0] = differ_distribution[0] * total_sum
        need_transition[1] = differ_distribution[1] * total_sum

        # state = np.concatenate([self_pos] + [self_vel] + entity_pos + other_pos + agent_land_alldis + differ_distribution)
        state = np.concatenate([self_pos.reshape(1, -1), self_vel.reshape(1, -1), np.array(entity_pos),
                                np.array(other_pos)] + agent_land_dis + [np.array(need_transition)] +
                               [np.array(world.target_distribute)] + [np.array(normalized_land)], axis=None)
        return state


    def reward(self, agent, world):
        """
        碰撞惩罚， 距离惩罚， 分布惩罚， 优先分布，然后距离，最后碰撞
        """
        collide_reward = 0
        dis_reward = 0
        kl_reward = 0
        differ_distribution = [0, 0]
        need_transition = [0, 0]

        land = [0, 0]  # 每个land上的agent数目
        for a in world.agents:
            dist = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.landmarks]
            min_index = dist.index(min(dist))
            land[min_index] += 1
        total_sum = sum(land)

        now_distribution = [value / total_sum for value in land]
        differ_distribution[0] = (world.target_distribute[0] - now_distribution[0])  # 调整状态
        differ_distribution[1] = (world.target_distribute[1] - now_distribution[1])
        need_transition[0] = differ_distribution[0] * total_sum
        need_transition[1] = differ_distribution[1] * total_sum

        kl_reward = -(abs(need_transition[0]) + abs(need_transition[1])) * 3
        # kl_divergence = entropy(now_distribution, target_distribute, 2)  # 计算两个离散型随机分布的散度
        # kl_reward = -kl_divergence * 20  # 通过调参实现奖励的区分

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):  # 有个bug，自己和自己发生碰撞，但不影响
                    collide_reward -= 1

        for a in world.agents:
            dist = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.landmarks]
            dis_reward -= min(dist)
        dis_reward = dis_reward * 5

        reward = kl_reward + collide_reward + dis_reward
        # print("target:{}".format(world.target_distribute))
        # print("kl_reward:{}, collide_reward:{}, dis_reward:{}".format(kl_reward, collide_reward, dis_reward))
        return reward

    def get_done(self, agent, world):
        """
        当达到第一个目标之后，换到第二个目标，依次轮流
        """
        return False






