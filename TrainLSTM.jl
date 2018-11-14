using JLD

TrainingData = CSV.read("TrainingData.csv";header = true, delim = ",")
TrainingData2 = AddMeanFeature(TrainingData,4,100)
rename!(TrainingData2, :Temp => :MeanX)

TrainingData3 = AddMeanFeature(TrainingData2,5,100)
rename!(TrainingData3, :Temp => :MeanY)

TrainingData4 = AddVarFeature(TrainingData3,4,100)
rename!(TrainingData4, :Temp => :VarX)

TData,Clasis = CreateTraining(TrainingData4[1:207000,:],100,[4,5,6,10,11,12],[8])

Params = initialiseLSTM(100,100,5,3)
TP, TN, FP, FN = 0,0,0,0
lstm_mems = Dict()
kyz = collect(keys(Params))
for (n, f) in enumerate(kyz)
    a,b,c = size(Params[f])
    lstm_mems[f] = zeros(a,b,c)
end
CurrentOrder = BootstrapDat(TData,6,100)
Data = CreateDataArray(CurrentOrder,TData)
Correct = CreateCorrectArray(CurrentOrder,TData)
#DataNorm = MinMaxNormalise(Data)
for i = 1:100
    Fwh,Fwcache = LSTMForwardPass(Data[1:6],Params)
    the,Clas,Afcache,preds = LSTMAfflineFW(Fwh,Params["U"],Params["b2"])
    #TP, TN, FP, FN = RightWrong(preds,Correct[1:6],TP,TN,FP,FN)
    loss,dtheta = softmaxloss(the,Correct[1:6])
    dh,dU,db2 = LSTMAfflineBW(dtheta,Afcache)
    all_grads = LSTMBackwardProp(dh,Fwcache,Params)
    all_grads["U"] = dU
    all_grads["b2"] = db2
    #lstm_mems = Dict()
    kys = collect(keys(all_grads))
    #for (n, f) in enumerate(kys)
    #a,b,c = size(Params[f])
    #lstm_mems[f] = zeros(a,b,c)
    #end
    for (n, k) in enumerate(kys)
    #all_grads[k] = clamp.(all_grads[k],-0.1,0.1)
    lstm_mems[k] += dot.(all_grads[k],all_grads[k])
    Params[k] += - (LR * all_grads[k]) ./ (sqrt.(lstm_mems[k]) .+ 1e-8)
    end
end

TrainedModel3 = load("TrainedWeights3.jld")
mems3 = load("Mems3.jld")

TrainedModel4,mems4 = TrainLSTM(TData,100,150,6,5,[6,7,8],50,0.1,TrainedModel4,mems4)
save("TrainedWeights3.jld", TrainedModel3)
save("Mems3.jld",mems3)

##standardscalar
#-1,1 normalisation
#features including mean and variance
#z against min max
#kaiser filtering
#compare average to max pool to consensus
