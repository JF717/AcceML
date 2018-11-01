TrainingData = CSV.read("TrainingData.csv";header = true, delim = ",")

temp = CreateTraining(TrainingData,features,correct)
