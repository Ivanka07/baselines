import gym
import numpy as np


"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []
vectorize_observation = False

def main():
    env = gym.make('FetchDrawTriangle-v1')
    numItr = 100
    initStateSpace = "random"
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)


    fileName = "data_fetch_draw"
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

    achieved_goals = np.array(lastObs['achieved_goal'])
    achieved_goals = achieved_goals.reshape((9,3))
    desired_goals  = np.array(lastObs['desired_goal'])
    desired_goals = desired_goals.reshape((9,3))
    
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    dginx = 0

    timeStep = 0 #count the total number of timesteps
    if vectorize_observation:
        episodeObs.append(vectorize_obs(lastObs))
    else:
        episodeObs.append(lastObs)
    cur_grip_pos = lastObs['observation'][0:3]
    while timeStep <= env._max_episode_steps:
        
        timeStep += 1
       # env.render()
        
        #print(cur_grip_pos)
        a = (desired_goals[dginx,:] - cur_grip_pos) 
        print(a)
        action = [a[0], a[1], a[2], 0.15]
        obsDataNew, reward, done, info = env.step(action)
        
        cur_grip_pos = obsDataNew['observation'][0:3]

        achieved_goals = np.array(obsDataNew['achieved_goal'])
        achieved_goals = achieved_goals.reshape((9,3))
        desired_goals  = np.array(obsDataNew['desired_goal'])
        desired_goals = desired_goals.reshape((9,3))
        
        #print(achieved_goals)
        print(desired_goals[dginx,:])
        ag = achieved_goals[dginx,:]
        dg = desired_goals[dginx,:]

        dist = np.linalg.norm(np.array(ag) - np.array(dg), axis=-1)
        
        if (dist < 0.05) and dginx < 8 :
            dginx +=1

        if timeStep < env._max_episode_steps:
            if vectorize_observation:
                episodeObs.append(vectorize_obs(obsDataNew))
            else:
                episodeObs.append(obsDataNew)



        episodeAcs.append(action)
        episodeInfo.append(info)

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
