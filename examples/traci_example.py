
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
cmd = ['sumo-gui', 
  '--net-file', 'simple/traffix2.net.xml', 
  '--route-files', 'simple/traffic.rou.xml',
  '--additional-files', 'simple/traffic.add.xml',
  '--end', '500']
actions = ['rGrG','ryry','GrGr','yryr']
def random_action():
	f = random.choice(actions)
	print "\naction chosen \n"
	print f
	return f

time_start = time()
north = [ ]
northSpeed=[]

for i in tqdm(range(5)):
	traci.start(cmd)
	for j in range(50):
		north = [ ]
		northSpeed=[]
		for i in range(0,75):
			north.append(0)
			northSpeed.append(0)
		print "\n\n\n NEXT ITERATION \n\n\n"
		traci.trafficlights.setRedYellowGreenState("0", random_action())
		traci.simulationStep()
		#positions = [traci.vehicle.getPosition(id) for id in traci.vehicle.getIDList()]
		speeds = [traci.vehicle.getSpeed(id) for id in traci.vehicle.getIDList()]
		for id in traci.vehicle.getIDList():
			print "\n"
			print id
			print traci.vehicle.getLaneID(id)
			if (traci.vehicle.getLaneID(id) =="n_0_0"):
			    pos =  traci.vehicle.getPosition(id)[1]-251
			    n1 = int((400-250)/traci.vehicle.getLength(id))
			    n2 = int(pos/traci.vehicle.getLength(id))
			    print "\n total"
			    print n1
			    print " ,"
			    print pos
			    print " position"
			    print n2
			    print "\n"
			    north[n2]=1
			    northSpeed[n2]= traci.vehicle.getSpeed(id)
			#print traci.vehicle.getPosition(id)
			
			#print traci.vehicle.getSpeed(id)
			print "\n"
		maxm = max(speeds)
		if(maxm == 0):
			maxm=1
		newList = [x / maxm for x in speeds]
		print "\n North position\n"
		print north
		print "\n North speed\n"
		print northSpeed
		
		
	traci.close()
time_end = time()
print "traci time elapsed: {}".format(time_end-time_start)
