TrainingData = CSV.read("TrainingData.csv";header = true, delim = ",")

TData,Clasis = CreateTraining(TrainingData[1:207000,:],100,[6,7,8],[10])
incor = []
for i = 600:600:nrow(TrainingData)
    if TrainingData[i,9] != TrainingData[i-599,9]
        push!(incor,i)
    end
end
Params = initialiseLSTM(100,100,5,3)

CurrentOrder = BootstrapDat(TData,6,100)
Data = CreateDataArray(CurrentOrder,TData)
Correct = CreateCorrectArray(CurrentOrder,TData)
#for i = 1:10
    Fwh,Fwcache = LSTMForwardPass(Data[1:6],Params)
    the,Clas,Afcache,preds = LSTMAfflineFW(Fwh,Params["U"],Params["b2"])
    #TP, TN, FP, FN = RightWrong(preds,Correct[1:6],TP,TN,FP,FN)
    loss,dtheta = softmaxloss(the,Correct[1:6])
    dh,dU,db2 = LSTMAfflineBW(dtheta,Afcache)
    all_grads = LSTMBackwardProp(dh,Fwcache,Params)
    all_grads["U"] = -dU
    all_grads["b2"] = db2
    lstm_mems = Dict()
    kys = collect(keys(all_grads))
    for (n, f) in enumerate(kys)
        a,b,c = size(Params[f])
        lstm_mems[f] = zeros(a,b,c)
    end
    for (n, k) in enumerate(kys)
        all_grads[k] = clamp.(all_grads[k],-0.1,0.1)
        lstm_mems[k] += dot.(all_grads[k],all_grads[k])
        Params[k] += - (LR * (all_grads[k] ./ (sqrt.(lstm_mems[k]) .+ 1e-8)))
    end
#end

TrainedModel = TrainLSTM(TData,100,200,6,5,[6,7,8],1,1)
