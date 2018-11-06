TrainingData = CSV.read("TrainingData.csv";header = true, delim = ",")

TData,Clasis = CreateTraining(TrainingData[1:207000,:],100,[6,7,8],[10])
incor = []
for i = 600:600:nrow(TrainingData)
    if TrainingData[i,9] != TrainingData[i-599,9]
        push!(incor,i)
    end
end
Params = initialiseLSTM(100,400,5,3)

CurrentOrder = BootstrapDat(TData,6,100)
Data = CreateDataArray(CurrentOrder,TData)
Correct = CreateCorrectArray(CurrentOrder,TData)
Fwh,Fwcache = LSTMForwardPass(Data[1:6],Params)
the,Clas,Afcache,preds = LSTMAfflineFW(Fwh,Params["U"],Params["b2"])
TP, TN, FP, FN = RightWrong(preds,Correct[1:6],TP,TN,FP,FN)
dtheta,dh,dU,db2,loss = LSTMAfflineBW(the,Clas,Correct[1:6],Afcache)
all_grads = LSTMBackwardProp(dh,Fwcache,Params)


TrainedModel = TrainLSTM(TData,100,100,6,5,[6,7,8],10,0.1)
