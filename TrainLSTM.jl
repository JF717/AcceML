TrainingData = CSV.read("TrainingData.csv";header = true, delim = ",")

temp,clas = CreateTraining(TrainingData,100,features,correct)
incor = []
for i = 600:600:nrow(TrainingData)-
    if TrainingData[i,9] != TrainingData[i-599,9]
        push!(incor,i)
    end
end
