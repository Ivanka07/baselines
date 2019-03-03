import gym
import numpy as np


"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []

def main():
    env = gym.make('FetchReach-v1')
    numItr = 100
    initStateSpace = "random2"
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)


    fileName = "data_fetch_reeach"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"

    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file

def goToGoal(env, lastObs):

    goal = lastObs['desired_goal']
    achieved_goal = lastObs['achieved_goal']
    print('goal =', goal,  'achived_goal', achieved_goal)
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)

    while timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0]
        goal_diff =  goal - achieved_goal

        for i in range(len(goal_diff)):
            action[i] = goal_diff[i]

        action[len(action)-1] = 0.05 #open

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        achieved_goal = obsDataNew['achieved_goal']
        if timeStep >= env._max_episode_steps: break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)


if __name__ == "__main__":
    main()
