# Sanity Test for Algorithms
import gym
import torch


def cartpole(algorithm, num_episodes=1000, render=False):
    import gym

    env = gym.make('CartPole-v1')
    score = 0.0
    #print_interval = 20
    print_interval = 10

    # Instantiate with Defaults
    agent = algorithm()

    for n_epi in range(num_episodes):
        s = env.reset()
        done = False

        while not done: # CartPole-v1 forced to terminates at 500 step.
            if render:
                env.render() #human viz of training process

            # Give Policy Observation, Get the Sampled Action -- ORIGINAL VERSION
            # prob = pi(torch.from_numpy(s).float())
            # m = Categorical(prob) #input action probabilities into Categorical Dist, sample discrete action
            # a = m.sample() #sample discrete action, i.e an int tensor

            # print("probs:", prob)
            # print("action:", a)
            # print("actionitem:", a.item())
            # s_prime, r, done, info = env.step(a.item())
            # pi.append_data(r,prob[a]))

            # Get Action from Logits -- CKG Updated
            obs = torch.from_numpy(s).float() #convert state given from Env into torch vec for passing to network (observation)
            logits = agent(obs) #forward pass, computes action logits
            action, prob_a = agent.sd_sampler(logits) #sample action (int) and get corresponding probabilties

            s_prime, r, done, info = env.step(action) #interact w environment and get next state info
            agent.append_data((r, prob_a)) #append the (reward, action_probability) tuple to the Policy Data Buffer
            s = s_prime #next/new state is now the current state
            score += r #episode reward

        agent.train() #after termination of episode, update the policy

        # Print Info per N Trajectories (episodes of running a specific Policy)
        if n_epi%print_interval==0 and n_epi!=0:
            print("Episode {}\tavg_score {}".format(n_epi, score/print_interval)) #original print statement
            #save_agent(n_epi, pi.state_dict(), score/print_interval)
            score = 0.0
    env.close()


