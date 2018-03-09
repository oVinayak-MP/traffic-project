import numpy as np
import os, sys
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:   
     sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
from tqdm import tqdm
import random
from time import time
pos1=33
pos2=48
actid=[]
def getSpeedandPosition(traci,laneid,avgvl,maxspeed,debug): #This function returns speed and position for the give lane it also should contains average vehicle length maxspeed and debug
    lanepos=[0]*100   #can be calculated from lane length and avgerage vehicle length
    lanespeed=[0]*100
   
    for id in traci.lane.getLastStepVehicleIDs(laneid):
	
        pos=traci.vehicle.getLanePosition(id)
        pos=int(pos/avgvl)
        lanepos[pos]=1
        lanespeed[pos]=traci.vehicle.getSpeed(id)
        
        if (debug == 1):
                print "index:" + str(pos)
                print "speed:" + str(lanespeed[pos])    
        
    lanespeed[:]=[round(x/maxspeed,2) for x in lanespeed]
    return lanespeed,lanepos
cmd = ['sumo-gui', '-c','sumo-map.sumocfg','--waiting-time-memory','10','-e','500']
#['rGrG','ryry','GrGr','yryr']
actions = [
"yrrrrrrrrrrrrrGG",
"rrrrrrrrrrrrGGrr",
"GGyrrrrrrrrrrrrr",
"rGGyyrrrrrrrrrrr",
"rrrGGyyrrrrrrrrr",
"rrrrrGGyrrrrrrrr",
"rrrrrrGGyrrrrrrr",
"rrrrrrrGGyrrrrrr",
"rrrrrrrrGGyyrrrr",
"rrrrrrrrrrGGrrrr",]
def random_action():
	f = random.choice(actions)
	print "\naction chosen \n"
	print f
	return f
def _cumulated_wt():
	cwt=0
	c=0
	for id in traci.vehicle.getIDList():
		cwt+=traci.vehicle.getWaitingTime(id)	#traci.vehicle.getAccumulatedWaitingTime(id)
		c+=traci.vehicle.getWaitingTime(id)
	print "wt:"+str(c)
	return cwt
#defn___
time_start = time()
avgvl=5
debug=0
laneid=['1to5_0', '1to5_1', '1to5_2', '2to5_0', '2to5_1', '2to5_2', '3to5_0', '3to5_1', '3to5_2', '4to5_0', '4to5_1', '4to5_2']

for i in tqdm(range(5)):
	traci.start(cmd)
	for j in range(100):
		for i in range(0,15):
			cwt1=_cumulated_wt();
			print "\n\n\n NEXT ITERATION \n\n\n"
			RanAct=random_action()
			traci.trafficlights.setRedYellowGreenState("5", RanAct)
			traci.simulationStep()
			print "lane_1"
			lane10speed,lane10pos=getSpeedandPosition(traci,laneid[0],avgvl,traci.lane.getMaxSpeed(laneid[0]),debug)
			lane11speed,lane11pos=getSpeedandPosition(traci,laneid[1],avgvl,traci.lane.getMaxSpeed(laneid[1]),debug)
			lane12speed,lane12pos=getSpeedandPosition(traci,laneid[2],avgvl,traci.lane.getMaxSpeed(laneid[2]),debug)
			lane1pos=np.vstack((np.asarray(lane10pos[33:48]),np.asarray(lane11pos[33:48]),np.asarray(lane12pos[33:48])))
		   	lane1speed=np.vstack((np.asarray(lane10speed[33:48]),np.asarray(lane11speed[33:48]),np.asarray(lane12speed[33:48])))
			print lane1pos,lane1speed
			print "lane_2"
			lane20speed,lane20pos=getSpeedandPosition(traci,laneid[3],avgvl,traci.lane.getMaxSpeed(laneid[3]),debug)
			lane21speed,lane21pos=getSpeedandPosition(traci,laneid[4],avgvl,traci.lane.getMaxSpeed(laneid[4]),debug)
			lane22speed,lane22pos=getSpeedandPosition(traci,laneid[5],avgvl,traci.lane.getMaxSpeed(laneid[5]),debug)
			lane2pos=np.vstack((np.asarray(lane20pos[33:48]),np.asarray(lane21pos[33:48]),np.asarray(lane22pos[33:48])))
		   	lane2speed=np.vstack((np.asarray(lane20speed[33:48]),np.asarray(lane21speed[33:48]),np.asarray(lane22speed[33:48])))
			print lane2pos,lane2speed
			print "lane_3"
			lane30speed,lane30pos=getSpeedandPosition(traci,laneid[6],avgvl,traci.lane.getMaxSpeed(laneid[6]),debug)
			lane31speed,lane31pos=getSpeedandPosition(traci,laneid[7],avgvl,traci.lane.getMaxSpeed(laneid[7]),debug)
			lane32speed,lane32pos=getSpeedandPosition(traci,laneid[8],avgvl,traci.lane.getMaxSpeed(laneid[8]),debug)
			lane3pos=np.vstack((np.asarray(lane30pos[33:48]),np.asarray(lane31pos[33:48]),np.asarray(lane32pos[33:48])))
		   	lane3speed=np.vstack((np.asarray(lane30speed[33:48]),np.asarray(lane31speed[33:48]),np.asarray(lane32speed[33:48])))
			print lane3pos,lane3speed
			print "lane_4"
			lane40speed,lane40pos=getSpeedandPosition(traci,laneid[9],avgvl,traci.lane.getMaxSpeed(laneid[9]),debug)
			lane41speed,lane41pos=getSpeedandPosition(traci,laneid[10],avgvl,traci.lane.getMaxSpeed(laneid[10]),debug)
			lane42speed,lane42pos=getSpeedandPosition(traci,laneid[11],avgvl,traci.lane.getMaxSpeed(laneid[11]),debug)
			lane4pos=np.vstack((np.asarray(lane40pos[33:48]),np.asarray(lane41pos[33:48]),np.asarray(lane42pos[33:48])))
		   	lane4speed=np.vstack((np.asarray(lane40speed[33:48]),np.asarray(lane41speed[33:48]),np.asarray(lane42speed[33:48])))
			print lane4pos,lane4speed
			cwt2=_cumulated_wt();
			reward = cwt1-cwt2
			print "\n Reward for the action is " + str(reward)
			print "\n earlier : "+str(cwt1)+"now : "+str(cwt2)
			
			
	traci.close()
time_end = time()

print "traci time elapsed: {}".format(time_end-time_start)
