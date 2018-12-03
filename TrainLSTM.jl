using JLD
using DSP
using Plots
#use a  Highpass low order Butterworth filter to filter with low delay
responsetype = Highpass(1; fs=10)
designmethod = Butterworth(1)


TrainingData = CSV.read("TrainingData.csv";header = true, delim = ",")
TrainingData[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},TrainingData[4]))
TrainingData[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},TrainingData[5]))
TrainingData[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},TrainingData[6]))

TrainingData[4] = clamp.(TrainingData[10],-25,25)
TrainingData[5] = clamp.(TrainingData[11],-25,25)
TrainingData[6] = clamp.(TrainingData[12],-25,25)
#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(TrainingData[10])-100
    TrainingData[10][i:i+2] .= mean(TrainingData[10][i+3:i+99])
    TrainingData[11][i:i+2] .= mean(TrainingData[11][i+3:i+99])
    TrainingData[12][i:i+2] .= mean(TrainingData[12][i+3:i+99])
end


#TrainingData2 = AddMeanFeature(TrainingData,4,100)
#rename!(TrainingData2, :Temp => :MeanX)

#TrainingData3 = AddMeanFeature(TrainingData2,5,100)
#rename!(TrainingData3, :Temp => :MeanY)

#TrainingData4 = AddVarFeature(TrainingData3,4,100)
#rename!(TrainingData4, :Temp => :VarX)

TrainingData[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),TrainingData[4],TrainingData[5],TrainingData[6])
TrainingData[:TotalG] = map((x,y,z) -> (x + y + z + 9.8),TrainingData[4],TrainingData[5],TrainingData[6])
TrainingData[:AbsX] = map((x) -> abs.(x),TrainingData[4])
TrainingData[:AbsY] = map((x) -> abs.(x),TrainingData[5])
TrainingData[:AbsZ] = map((x) -> abs.(x),TrainingData[6])

TrainingData[:TotalG] = map((x,y,z) -> (x + y + z),TrainingData[4],TrainingData[5],TrainingData[6])
TData,Clasis = CreateTraining(TrainingData,100,[4,5,6,10,11,12,13,14],[8])

Params = initialiseLSTM(100,100,5,8)
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
#for i = 1:100
Fwh,Fwcache = LSTMForwardPass(Data[1:6],Params)
the,Afcache,preds = LSTMAfflineFW2(Fwh,Params["U"],Params["U2"],Params["b2"],Params["b3"])
#    #TP, TN, FP, FN = RightWrong(preds,Correct[1:6],TP,TN,FP,FN)
loss,dtheta = softmaxloss2(the,Correct[1:6])
dh,dh2,dU,dU2,db2,db3 = LSTMAfflineBW2(dtheta,Afcache)
#    all_grads = LSTMBackwardProp(dh,Fwcache,Params)
#    all_grads["U"] = dU
#    all_grads["b2"] = db2
    #lstm_mems = Dict()
#    kys = collect(keys(all_grads))
#    #for (n, f) in enumerate(kys)
    #a,b,c = size(Params[f])
    #lstm_mems[f] = zeros(a,b,c)
    #end
#    for (n, k) in enumerate(kys)
    #all_grads[k] = clamp.(all_grads[k],-0.1,0.1)
#    lstm_mems[k] += dot.(all_grads[k],all_grads[k])
#    Params[k] += - (LR * all_grads[k]) ./ (sqrt.(lstm_mems[k]) .+ 1e-8)
#    end
#end

TrainedModel11 = load("TrainedModel11.jld")
mems11 = load("Mems11.jld")

TrainedModel13,mems13,CM3 = TrainLSTM(TData,100,100,6,5,[4,5,6,10,11,12,13,14],1,0.1,TrainedModel13,mems13)

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

#filtered all 3 + filtered combined.
save("TrainedModel7.jld", TrainedModel7)
save("mems7.jld",mems7)

#filtered and then the delay removed for all 3 + total
save("TrainedModel11.jld", TrainedModel11)
save("mems11.jld",mems11)


Collar10 = CSV.read("Collar10AccelCor.csv";header = true, delim = ",")
