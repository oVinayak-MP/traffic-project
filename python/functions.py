def getSpeedandPosition(traci,laneid,avgvl,maxspeed,debug): #This function returns speed and position for the give lane it also should contains average vehicle length
    lanepos=[0]*100
    lanespeed=[0]*100
    
    for id in traci.lane.getLastStepVehicleIDs(laneid):
        speed=traci.vehicle.getSpeed(id)
        pos=traci.vehicle.getLanePosition(id)
        pos=int(pos/avgvl)
        
        lanepos[pos]=1
        lanespeed[pos]=traci.vehicle.getSpeed(id)
        
        if (debug == 1):
                print "index:" + str(pos)
                print "speed:" + str(lanespeed[pos])
                
        
    lanespeed[:]=[round(x/maxspeed,2) for x in lanespeed]
    return lanespeed,lanepos
