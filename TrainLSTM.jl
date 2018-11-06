TrainingData = CSV.read("TrainingData.csv";header = true, delim = ",")

temp,clas = CreateTraining(TrainingData,100,features,correct)
incor = []
for i = 600:600:nrow(TrainingData)-
    if TrainingData[i,9] != TrainingData[i-599,9]
        push!(incor,i)
    end
end

p = initialiseLSTM(100,400,5,3)

h1,s1,c1 = LSTMForward(x_t,Params["h_0"],Params["s_0"],Params)
h1, cd1 = LSTMForwardPass(x,Params)

th1, y1, ca1,pred1 = LSTMAfflineFW(h1,Params["U"],Params["b2"])
dth1,dh1,du1,db21,loss1 = LSTMAfflineBW(th1,y1,yt,ca1)
