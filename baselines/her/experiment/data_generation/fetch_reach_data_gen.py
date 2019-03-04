import gym
import numpy as np


"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []
vectorize_observation = True

def main():
    env = gym.make('FetchReach-v1')
    numItr = 200
    initStateSpace = "random"
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)


    fileName = "data_fetch_reach"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"

    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file
    return fileName

def vectorize_obs(obs):
    vect_obs = []
    for k,v in obs.items():
        for element in v:
            vect_obs.append(element)
    return vect_obs
            
def goToGoal(env, lastObs):

    goal = lastObs['desired_goal']
    achieved_goal = lastObs['achieved_goal']
    print('goal =', goal,  'achived_goal', achieved_goal)
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    timeStep = 0 #count the total number of timesteps
    if vectorize_observation:
        episodeObs.append(vectorize_obs(lastObs))
    else:
        episodeObs.append(lastObs)

    while timeStep <= env._max_episode_steps:
        
        timeStep += 1
        #env.render()
        action = [0, 0, 0, 0]
        goal_diff =  goal - achieved_goal

        for i in range(len(goal_diff)):
            action[i] = goal_diff[i]

        action[len(action)-1] = 0.05 #open
        obsDataNew, reward, done, info = env.step(action)
        if timeStep < env._max_episode_steps:
            if vectorize_observation:
                episodeObs.append(vectorize_obs(obsDataNew))
            else:
                episodeObs.append(obsDataNew)



        episodeAcs.append(action)
        episodeInfo.append(info)


        achieved_goal = obsDataNew['achieved_goal']
        if timeStep >= env._max_episode_steps: break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)

def test(filename):
    print('Test')
    data = np.load(fileName)
    print('Obs shape={}'.format(data['obs'].shape))
    print('Acs shape={}'.format(data['acs'].shape))
    print('Infos shape={}'.format(data['info'].shape))

if __name__ == "__main__":
    fileName = main()
    test(fileName)
