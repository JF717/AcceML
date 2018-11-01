TrainingData = CSV.read("TrainingData.csv";header = true, delim = ",")

temp,clas = CreateTraining(TrainingData,100,features,correct)
