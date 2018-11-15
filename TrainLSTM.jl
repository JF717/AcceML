using JLD
using DSP

responsetype = Highpass(1; fs=10)
designmethod = Butterworth(4)
filt(digitalfilter(responsetype, designmethod),)

TrainingData = CSV.read("TrainingData.csv";header = true, delim = ",")
TrainingData2 = AddMeanFeature(TrainingData,4,100)
rename!(TrainingData2, :Temp => :MeanX)

TrainingData3 = AddMeanFeature(TrainingData2,5,100)
rename!(TrainingData3, :Temp => :MeanY)

TrainingData4 = AddVarFeature(TrainingData3,4,100)
rename!(TrainingData4, :Temp => :VarX)

TrainingData4[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),TrainingData4[4],TrainingData4[5],TrainingData4[6])
TrainingData4[:TotalG] = map((x,y,z) -> (x + y + z + 9.8),TrainingData4[4],TrainingData4[5],TrainingData4[6])
TData,Clasis = CreateTraining(TrainingData4[1:207000,:],100,[4,5,6,13,14,10,11,12],[8])

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

TrainedMode6,mems6 = TrainLSTM(TData,100,100,6,5,[4,5,6,13,14,10,11,12],100,0.1,TrainedMode6,mems6)

#all 3 raw + meanX meany and variancex got to same 60%
save("TrainedModel6feat.jld", TrainedModel6feat)
save("mems6feat.jld",mems6feat)
#3 normalised + absolutue total across all 3 got to 60% quicker
save("TrainedModelWithTot.jld", TrainedModelWithTot)
save("memsWithTot.jld",memsWithTot)
#3 normalised + abs tot and raw tot
save("TrainedModel5.jld", TrainedMode5)
save("mems5.jld",mems5)

#all 3 + abs tot and tot normalised + meanx meany and varx
save("TrainedMode6.jld", TrainedMode6)
save("mems6.jld",mems6)
##standardscalar
#-1,1 normalisation done
#features including mean and variance done
#z against min max
#kaiser filtering
#compare average to max pool to consensus done

#Wavelet desnoising
Collar10 = CSV.read("Collar10AccelCor.csv";header = true, delim = ",")
