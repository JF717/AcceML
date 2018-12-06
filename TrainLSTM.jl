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

TrainingData[:TotalG] = map((x,y,z) -> (x + y + z + 9.8),TrainingData[4],TrainingData[5],TrainingData[6])
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
all_grads = LSTMBackwardProp(dh,Fwcache,Params)
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

TrainedModel14,mems14,CM4 = TrainLSTM(TData,100,100,6,5,[4,5,6,10,11,12,13,14],1000,0.1,TrainedModel14,mems14)

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
save("TrainedModel14.jld", TrainedModel14)
save("mems14.jld",mems14)


Collar6 = CSV.read("Collar6AccelCor.csv";header = true, delim = ",")
Collar6[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar6[4]))
Collar6[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar6[5]))
Collar6[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar6[6]))

Collar6[4] = clamp.(Collar6[10],-25,25)
Collar6[5] = clamp.(Collar6[11],-25,25)
Collar6[6] = clamp.(Collar6[12],-25,25)
#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar6[7])-100
    Collar6[7][i:i+2] .= mean(Collar6[7][i+3:i+99])
    Collar6[8][i:i+2] .= mean(Collar6[8][i+3:i+99])
    Collar6[9][i:i+2] .= mean(Collar6[9][i+3:i+99])
end


Collar6[:TotalG] = map((x,y,z) -> (x + y + z),Collar6[4],Collar6[5],Collar6[6])
Collar6[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar6[4],Collar6[5],Collar6[6])
Classed6 =  RunLSTM(Collar6,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed6[2])

Collar10 = CSV.read("Collar10AccelCor.csv";header = true, delim = ",")
Collar10[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar10[4]))
Collar10[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar10[5]))
Collar10[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar10[6]))

#Collar10[4] = clamp.(Collar10[10],-25,25)
#Collar10[5] = clamp.(Collar10[11],-25,25)
#Collar10[6] = clamp.(Collar10[12],-25,25)
#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar10[7])-100
    Collar10[7][i:i+2] .= mean(Collar10[7][i+3:i+99])
    Collar10[8][i:i+2] .= mean(Collar10[8][i+3:i+99])
    Collar10[9][i:i+2] .= mean(Collar10[9][i+3:i+99])
end


Collar10[:TotalG] = map((x,y,z) -> (x + y + z),Collar10[4],Collar10[5],Collar10[6])
Collar10[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar10[4],Collar10[5],Collar10[6])
Classed10 =  RunLSTM(Collar10,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed10[2])

Collar5 = CSV.read("Collar5AccelCor.csv";header = true, delim = ",")
Collar5[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar5[4]))
Collar5[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar5[5]))
Collar5[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar5[6]))

#Collar10[4] = clamp.(Collar10[10],-25,25)
#Collar10[5] = clamp.(Collar10[11],-25,25)
#Collar10[6] = clamp.(Collar10[12],-25,25)
#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar5[7])-100
    Collar5[7][i:i+2] .= mean(Collar5[7][i+3:i+99])
    Collar5[8][i:i+2] .= mean(Collar5[8][i+3:i+99])
    Collar5[9][i:i+2] .= mean(Collar5[9][i+3:i+99])
end


Collar5[:TotalG] = map((x,y,z) -> (x + y + z),Collar5[4],Collar5[5],Collar5[6])
Collar5[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar5[4],Collar5[5],Collar5[6])
Classed5 =  RunLSTM(Collar5,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed5[2])

Collar1 = CSV.read("Collar1AccelCor.csv";header = true, delim = ",")
Collar1[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar1[4]))
Collar1[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar1[5]))
Collar1[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar1[6]))

#Collar10[4] = clamp.(Collar10[10],-25,25)
#Collar10[5] = clamp.(Collar10[11],-25,25)
#Collar10[6] = clamp.(Collar10[12],-25,25)
#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar1[7])-100
    Collar1[7][i:i+2] .= mean(Collar1[7][i+3:i+99])
    Collar1[8][i:i+2] .= mean(Collar1[8][i+3:i+99])
    Collar1[9][i:i+2] .= mean(Collar1[9][i+3:i+99])
end


Collar1[:TotalG] = map((x,y,z) -> (x + y + z),Collar1[4],Collar1[5],Collar1[6])
Collar1[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar1[4],Collar1[5],Collar1[6])
Classed1 =  RunLSTM(Collar1,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed1[2])

Collar2 = CSV.read("Collar2AccelCor.csv";header = true, delim = ",")
Collar2[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar2[4]))
Collar2[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar2[5]))
Collar2[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar2[6]))

#Collar10[4] = clamp.(Collar10[10],-25,25)
#Collar10[5] = clamp.(Collar10[11],-25,25)
#Collar10[6] = clamp.(Collar10[12],-25,25)
#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar2[7])-100
    Collar2[7][i:i+2] .= mean(Collar2[7][i+3:i+99])
    Collar2[8][i:i+2] .= mean(Collar2[8][i+3:i+99])
    Collar2[9][i:i+2] .= mean(Collar2[9][i+3:i+99])
end


Collar2[:TotalG] = map((x,y,z) -> (x + y + z),Collar2[4],Collar2[5],Collar2[6])
Collar2[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar2[4],Collar2[5],Collar2[6])
Classed2 =  RunLSTM(Collar2,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed2[2])
