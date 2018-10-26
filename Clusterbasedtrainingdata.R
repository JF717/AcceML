###file to use clusters to build trainingdataset

setwd("~/GitHub/UnsupCollar/DTWKmeansAccel")

#lying down. easy to recognise it is the most common cluster and characterised by a near 0
#measurement in the x axis.
liedowndat = data.frame()
for (j in (seq(1,15))){
  
  #print(j)
  filename <- paste("Collar",j,"AccelCor.csv", sep = "") 
  if(file.exists(filename)){
    #print("File Exists")
    assign(paste("Collar",j,"Accel" ,sep = ""), read.csv(filename, na.strings=c('NA'), header = T ))
    assinam <- paste("Collar",j,"Assignments.csv",sep="")
    assign(paste("AssignmentsCollar",j,sep=""), read.csv(assinam, header = T))
    
    curcol <- paste("Collar",j,"Accel" ,sep = "")
    curcol <- get(curcol)
    curcol$Collar <- paste("Collar",j,sep = "")
    
    curassi <- paste("AssignmentsCollar",j,sep="")
    curassi <- get(curassi)
    
    
    numdat = 50
    longestclust = 0
    longlen = 0
    for (i in c(2:length(curassi))){
      clustlen = length(curassi[!is.na(curassi[i]),i])
      #print(clustlen)
      if(clustlen > longlen){
        longestclust = i
        longlen = clustlen
        #print(paste("longest cluster is", longestclust))
      }
    
    }
    
    #liedownlist = list()
    tdat = data.frame()
    iter = 1
    while(i < numdat){
      #print(i)
      ind = curassi[sample(nrow(curassi), 1,replace = F),longestclust]
      if (ind < 300 | ind > (nrow(curcol) - 300)){
      next}
      extr = c((ind-301):(ind+299))
      #print(extr)
      if(abs(max(curcol[extr,4]) - min(curcol[extr,4]))>1){
        #print(abs(max(curcol[extr,4]) - min(curcol[extr,4])))
        next
      }else{ tdat = curcol[extr,]
      liedowndat = rbind(liedowndat,tdat)
        #liedownlist[[i]] <- curcol[extr,] 
            i = i + 1}
      
      if(i == numdat){
        break
        #print("Hopefully appending")
        #for (x in c(1:length(liedownlist))){
       # liedowndat <- rbind(liedowndat,liedownlist[[x]])}
        #rm(liedownlist)
      }
      iter = iter +1
     #print(iter)
      if(iter > 500){break}}
  }
}

# running training data

#first we will search the other clusters to see if it is running or what it is based
#on features from our data and the videos


# running is noticable because the x is away from 0 and there is high variation in all axes.
rundat = data.frame()

for (j in seq(1,15)){
  
  curcol <- paste("Collar",j,"Accel" ,sep = "")
  if(!exists(curcol)){next}
  curcol <- get(curcol)
  curcol$Collar <- paste("Collar",j,sep = "")
  
  curassi <- paste("AssignmentsCollar",j,sep="")
  if(!exists(curassi)){next}
  curassi <- get(curassi)
  
  
    for (i in c(2:length(curassi))){
      clustlen = length(curassi[!is.na(curassi[i]),i])
      #print(paste("Cluster",i))
      if(clustlen != nrow(curassi)){
        #print("not lying down")
        numsamp = 0
        iter = 0
        while(numsamp != 25){
        ind = curassi[sample(clustlen, 1),i]
        if (ind < 100 | ind > (nrow(curcol)-100)){iter = iter +1 
        next}
        #print(ind)
        tdat = curcol[ind:(ind+99),]
        if(mean(abs(tdat[,4])) > 1 & abs(max(tdat[,4]) - min(tdat[,4])) > 12){
          #print("It is running")
         rundat = rbind(rundat,tdat)
         numsamp = numsamp + 1
        }
        iter = iter + 1
        if(iter == clustlen){break}
        
      }}
      
    }
}

#great looks like that worked time to move on to walking.

#walking is trickier as it is very much like foraging the angle of the head being
#the key difference

#walking is generally noticable when there is a movement in the x axis so not close to 0 
#but there is not a high variation like with running. 


walkdat = data.frame()

for (j in seq(1,15)){
  
  curcol <- paste("Collar",j,"Accel" ,sep = "")
  if(!exists(curcol)){next}
  curcol <- get(curcol)
  curcol$Collar <- paste("Collar",j,sep = "")
  
  curassi <- paste("AssignmentsCollar",j,sep="")
  if(!exists(curassi)){next}
  curassi <- get(curassi)
  
  
  for (i in c(2:length(curassi))){
    clustlen = length(curassi[!is.na(curassi[i]),i])
   # print(paste("Cluster",i))
    if(clustlen != nrow(curassi)){
      #print("not lying down")
      numsamp = 0
      iter = 0
      while(numsamp != 25){
        ind = curassi[sample(clustlen, 1),i]
        if (ind < 100 | ind > (nrow(curcol)-100)){iter = iter +1 
        next}
        #print(ind)
        tdat = curcol[ind:(ind+99),]
        if(mean(abs(tdat[,4])) > 3 & abs(max(tdat[,4]) - min(tdat[,4])) < 10 & abs(mean(tdat[,5]) - mean(curcol[,5])) < 2){
          #print("It is walking")
          walkdat = rbind(walkdat,tdat)
          numsamp = numsamp + 1
        }
        iter = iter + 1
        if(iter == clustlen){break}
        
      }}
    
  }
}


#doing foraging, distinct from walking in that it generates a high signal in the y axis due
#to the position of the head being downwards towards the ground
foragedat = data.frame()

for (j in seq(1,15)){
  
  curcol <- paste("Collar",j,"Accel" ,sep = "")
  if(!exists(curcol)){next}
  curcol <- get(curcol)
  curcol$Collar <- paste("Collar",j,sep = "")
  
  curassi <- paste("AssignmentsCollar",j,sep="")
  if(!exists(curassi)){next}
  curassi <- get(curassi)
  
  
  for (i in c(2:length(curassi))){
    clustlen = length(curassi[!is.na(curassi[i]),i])
    # print(paste("Cluster",i))
    if(clustlen != nrow(curassi)){
      #print("not lying down")
      numsamp = 0
      iter = 0
      while(numsamp != 25){
        ind = curassi[sample(clustlen, 1),i]
        if (ind < 100 | ind > (nrow(curcol)-100)){iter = iter +1 
        next}
        #print(ind)
        tdat = curcol[ind:(ind+99),]
        if(mean(abs(tdat[,4])) > 3 & abs(max(tdat[,4]) - min(tdat[,4])) < 10 & abs(mean(tdat[,5]) - mean(curcol[,5])) > 5){
          #print("It is walking")
          foragedat = rbind(foragedat,tdat)
          numsamp = numsamp + 1
        }
        iter = iter + 1
        if(iter == clustlen){break}
        
      }}
    
  }
}


#going to try and seperate lyind down from sitting and standing, these also have some x but
#almost no variation within that x, with slight differences in the y and z,
#it would most likely be characterisable by being near to movement. 

sitdat = data.frame()

for (j in seq(1,15)){
  
  curcol <- paste("Collar",j,"Accel" ,sep = "")
  if(!exists(curcol)){next}
  curcol <- get(curcol)
  curcol$Collar <- paste("Collar",j,sep = "")
  
  curassi <- paste("AssignmentsCollar",j,sep="")
  if(!exists(curassi)){next}
  curassi <- get(curassi)
  
  
  for (i in c(2:length(curassi))){
    clustlen = length(curassi[!is.na(curassi[i]),i])
    # print(paste("Cluster",i))
    if(clustlen != nrow(curassi)){
      #print("not lying down")
      numsamp = 0
      iter = 0
      while(numsamp != 25){
        ind = curassi[sample(clustlen, 1),i]
        if (ind < 100 | ind > (nrow(curcol)-100)){iter = iter +1 
        next}
        #print(ind)
        tdat = curcol[ind:(ind+99),]
        if(abs(max(tdat[,4]) - min(tdat[,4])) < 2 & mean(abs(tdat[,4])) > 2 &  mean(abs(tdat[,4]< 6))){
          #print("It is walking")
          sitdat = rbind(sitdat,tdat)
          numsamp = numsamp + 1
        }
        iter = iter + 1
        if(iter == clustlen){break}
        
      }}
    
  }
}

#label the beahviour they are performing
foragedat$Behaviour <- "Forage"
liedowndat$Behaviour <- "Lie Down"
rundat$Behaviour <- "Run"
walkdat$Behaviour <- "Walk"
sitdat$Behaviour <- "Sit/Stand"

#if made for plotting get rid of row ID
#sitdat$rid <- NULL
#liedowndatdat$rid <- NULL
#walkdat$rid <- NULL
#foragedat$rid <- NULL
#rundat$rid <- NULL
#dropping the weird extra lie down we picked up
liedowndat2 <- liedowndat[1:(nrow(liedowndat)-41),]
#making the combined dataframe
combineddata = rbind(liedowndat2,sitdat)
combineddata = rbind(combineddata, walkdat)
combineddata = rbind(combineddata, foragedat)
combineddata = rbind(combineddata, rundat)

#trainingdata = data.frame()
#creating a unique ID for every timeperiod
combineddata$ID <- ""
for (i in seq(1,nrow(combineddata),by = 100)){
  combineddata$ID[seq(i,i+99)] <- paste(combineddata$X[i],combineddata$Collar[i],sep = "")
  
}

#checking for repeats
reps <- as.data.frame(table(combineddata$ID))
repID <- data.frame()
for (i in seq(1,nrow(reps))){
  if(reps$Freq[i] > 100){
    b <- data.frame(reps$Var1[i],reps$Freq[i]/100)
    repID<-rbind(repID,b)
    
  }
}

#finding those repeats
reploc <- which(combineddata$ID %in% repID$reps.Var1.i.)
reploc2 <- NULL
for (i in seq(1,length(reploc),by = 100)){
reploc2 <- c(reploc2,reploc[i])
}
#checking to see if there is repitition within behaviours or between.
repdat <- data.frame()
for (j in seq(1,nrow(combineddata))){
  for (i in reploc2){
    if(j == i){
      repdat <- rbind(repdat,combineddata[j,])
      }
  }
}
#order this to find out
repdat2 <- repdat[with(repdat,order(repdat$ID)),]

#we can see nearly all is repeated because of sampling multiple times we will remove these
#to visually inspect any differences and manually classify repeats that cross behaviours
difreps <- data.frame()
for (i in seq(1,(nrow(repdat2)-1))){
  if(repdat2$ID[i] == repdat2$ID[i+1]){
    #print(paste(repdat2$ID[i],repdat2$ID[i+1]))
    if(repdat2$Behaviour[i] != repdat2$Behaviour[i+1]){
     # print("Different")
      a <- rbind(repdat2[i,],repdat2[i+1,])
      #print(a)
      difreps <- rbind(difreps,a)
    }
  }
}

#ok so we have 128 doubles which means there are 64 times that a behaviour has been
#classified into something twice. This tends to be between sit / stand and others.
#we will plot them to take a look at them and decide what category to put them in.
plts <- list()
difrepsdata <- combineddata[which(combineddata$ID %in% difreps$ID),]
for (j in seq(1,(nrow(difrepsdata)-100),by = 100)){
  for (i in seq(1,nrow(difreps),by = 2)){
    if(difrepsdata$X[j] == difreps$X[i]){
      pltdat <- difrepsdata[j:(j+99),]
      print(pltdat)
     # plts[[(i/2)]] <- ggplot()+
       # theme_bw()+
       # geom_line(data =pltdat, aes(X, obj.X),color = "Blue")+
      #  geom_line(data =pltdat, aes(X, obj.Y),color = "Red")+
       # geom_line(data =pltdat, aes(X, obj.Z), color = "Green")+
       # ylim(c(-20,20))
    }
  }
}


#that was hard maybe come back to that. lets try removing them first so there is
#no confusion
combidatrmdup <- combineddata
for (i in seq(1,nrow(difreps),by = 2)){
  combidatrmdup <- subset(combidatrmdup, ID!=difreps$ID[i])
}

#now we want to find the reps that aren't two different behaviours and remove one of them
reps2 <- as.data.frame(table(combidatrmdup$ID))
repID2 <- data.frame()
for (i in seq(1,nrow(reps2))){
  if(reps2$Freq[i] > 100){
    b <- data.frame(reps2$Var1[i],reps2$Freq[i]/100)
    repID2<-rbind(repID2,b)
    
  }
}


#we know what is repeated now we just find the index of them and drop em
for (i in seq(1,nrow(repID2))){
  replocs <- which(combidatrmdup$ID %in% repID2[i,1])
  replocs <- replocs[101:length(replocs)]
  combidatrmdup <- combidatrmdup[-c(replocs),]
}


#now we've removed all our repeats problems we have a dataset of named examples but we need
#to turn this into sequential minibatches and remove bias of different sized behaviour sets

#first we will build a set of all the unique ID's that are in our remaining data. We will 
#then use this to check if the surrounding indexes exist.

IDforbatch <- combidatrmdup[!duplicated(combidatrmdup$ID),]
#now we have a dataframe of the first time we encountered our unique ID with it's index
#so we can now check to see if the indexes -300 +200 exist so we have a 60 second minibatch.

for (i in seq(1,nrow(IDforbatch))){

  b0 <- IDforbatch
  bn3 <- IDforbatch[which(IDforbatch$X %in% (IDforbatch$X[i] - 300)  & IDforbatch$Collar[i] == IDforbatch$Collar[i]),]
  bn2 <-IDforbatch[which(IDforbatch$X %in% (IDforbatch$X[i] - 200)  & IDforbatch$Collar[i] == IDforbatch$Collar[i]),]
  bn1 <-IDforbatch[which(IDforbatch$X %in% (IDforbatch$X[i] - 100)  & IDforbatch$Collar[i] == IDforbatch$Collar[i]),]
  b1<-IDforbatch[which(IDforbatch$X %in% (IDforbatch$X[i] + 100)  & IDforbatch$Collar[i] == IDforbatch$Collar[i]),]
  b2<-IDforbatch[which(IDforbatch$X %in% (IDforbatch$X[i] + 200)  & IDforbatch$Collar[i] == IDforbatch$Collar[i]),]
  
  
}