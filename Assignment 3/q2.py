import numpy as np
from drawFromADist import drawFromADist 
import random
from scipy.stats import norm
''' %the temporal context model assumes that the past becomes increasingly
% dissimilar to the future, so that memories become harder to retrieve the
% farther away in the past they are '''

def compdelta():
    x = [norm(0.2, 0.5), norm(5, 1)]
    c = np.random.choice([0, 1], p=[0.6, 0.4])
    return x[c].rvs()




def tcm():
    N_WORLD_FEATURES = 5
    N_ITEMS = 10
    ENCODING_TIME = 500
    TEST_TIME = 20

    '''% we are going to model the world as a set of N continuous-valued features.
    % we will model observations of states of the world as samples from N
    % Gaussians with time-varying means and fixed variance. For simplicity,
    % assume that agents change nothing in the world.

    % first fix the presentation schedule; I am assuming its random'''

 
    schedule = [2, 14, 25, 61, 153, 261, 269, 272, 462, 464] ## Fixed Schedule for which we get almost average success rate of 7
    schedule_load = ENCODING_TIME/np.median(np.diff(schedule))                ##% variable important for parts 2,3 of assignment
    encoding = np.zeros((N_ITEMS,N_WORLD_FEATURES+1))

    world_m = np.array([5, 3, 1, 5, 1]) ##Fixed initial world means for which we get nearly average success rate of 7
    
    world_var = 1
    beta_param = 0.001                ## % what does this parameter affect? Weight of new sampled world features values in new world state 
    m = 0

    world = np.array([0]*5)## Initial world state

    
    ##% simulating encoding

    for time in range(1,ENCODING_TIME+1):
        delta = compdelta()
        world_m = world_m + delta
        temp = world_m + world_var*np.random.randn(N_WORLD_FEATURES)
        prod = np.dot(temp,world)
        p = (1+(beta_param**2)*((prod**2)-1))**0.5 - beta_param*prod
        world = p*world + beta_param*temp
        ##% any item I want to encode in memory, I encode in association with the
        ##% state of the world at that time.
        if m<N_ITEMS :
            if(time==schedule[m]):
                encoding[m,:] = np.append(world,m)            ##% encode into the encoding vector
                m =  m + 1

    ##% simulating retrieval using SAM, but with a bijective image-item mapping


    out = [0]*TEST_TIME
    while(time<ENCODING_TIME+TEST_TIME):
        
    ##% the state of the world is the retrieval cue
        world_m = world_m + delta
        temp = world_m + world_var*np.random.randn(N_WORLD_FEATURES)
        prod = np.dot(temp,world)
        p = (1+(beta_param**2)*((prod**2)-1))**0.5 - beta_param*prod
        world = p*world + beta_param*temp
        soa = [0]*N_ITEMS
        for m in range(N_ITEMS):
        
            soa[m] = np.dot(encoding[m,:5], world)    ## % finding association strengths

        soa = soa/np.sum(soa)                                                                ## % normalize
        out[time-ENCODING_TIME] = np.where(drawFromADist(soa)==1)
       
        time = time + 1
    
    success= np.unique(out)     ##% success is number of unique retrievals

   
    return schedule_load, len(success)

ch = input('1. Single Trial \n2. Multiple Trials \n')

if(ch == '1'):
    schedule_load, nunique = tcm()
    print('No of unique retrivals :', nunique)
    print('Scheduling load : ',schedule_load)

elif(ch == '2'):
     n = int(input('No of trials :'))
     sch = []
     uniq= []
     for i in range(n):
         schedule_load, nunique = tcm()
         sch.append(schedule_load)
         uniq.append(nunique)
         
         
     print(schedule)
     print('Schedule Load :',np.mean(sch))
     print('Mean Unique Retrivals :', np.mean(uniq))
         
else:
    print('Invalid Choice')
'''% humans can retrieve about 7 items effectively from memory. get this model
% to behave like humans'''
