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
