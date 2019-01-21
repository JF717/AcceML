using JLD
using DSP
using Plots

responsetype = Highpass(1; fs=10)
designmethod = Butterworth(1)

TrainedModel14 = load("TrainedModel14.jld")


Collar6 = CSV.read("Collar6AccelCor.csv";header = true, delim = ",")
Collar6[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar6[4]))
Collar6[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar6[5]))
Collar6[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar6[6]))

#Collar6[4] = clamp.(Collar6[10],-25,25)
#Collar6[5] = clamp.(Collar6[11],-25,25)
#Collar6[6] = clamp.(Collar6[12],-25,25)
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

Collar3 = CSV.read("Collar3AccelCor.csv";header = true, delim = ",")
Collar3[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar3[4]))
Collar3[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar3[5]))
Collar3[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar3[6]))

#Collar10[4] = clamp.(Collar10[10],-25,25)
#Collar10[5] = clamp.(Collar10[11],-25,25)
#Collar10[6] = clamp.(Collar10[12],-25,25)
#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar3[7])-100
    Collar3[7][i:i+2] .= mean(Collar3[7][i+3:i+99])
    Collar3[8][i:i+2] .= mean(Collar3[8][i+3:i+99])
    Collar3[9][i:i+2] .= mean(Collar3[9][i+3:i+99])
end


Collar3[:TotalG] = map((x,y,z) -> (x + y + z),Collar3[4],Collar3[5],Collar3[6])
Collar3[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar3[4],Collar3[5],Collar3[6])
Classed3 =  RunLSTM(Collar3,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed3[2])


Collar4 = CSV.read("Collar4AccelCor.csv";header = true, delim = ",")
Collar4[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar4[4]))
Collar4[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar4[5]))
Collar4[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar4[6]))

#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar4[7])-100
    Collar4[7][i:i+2] .= mean(Collar4[7][i+3:i+99])
    Collar4[8][i:i+2] .= mean(Collar4[8][i+3:i+99])
    Collar4[9][i:i+2] .= mean(Collar4[9][i+3:i+99])
end


Collar4[:TotalG] = map((x,y,z) -> (x + y + z),Collar4[4],Collar4[5],Collar4[6])
Collar4[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar4[4],Collar4[5],Collar4[6])
Classed4 =  RunLSTM(Collar4,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed4[2])

Collar7 = CSV.read("Collar7AccelCor.csv";header = true, delim = ",")
Collar7[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar7[4]))
Collar7[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar7[5]))
Collar7[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar7[6]))

#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar7[7])-100
    Collar7[7][i:i+2] .= mean(Collar7[7][i+3:i+99])
    Collar7[8][i:i+2] .= mean(Collar7[8][i+3:i+99])
    Collar7[9][i:i+2] .= mean(Collar7[9][i+3:i+99])
end


Collar7[:TotalG] = map((x,y,z) -> (x + y + z),Collar7[4],Collar7[5],Collar7[6])
Collar7[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar7[4],Collar7[5],Collar7[6])
Classed7 =  RunLSTM(Collar7,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed7[2])

Collar8 = CSV.read("Collar8AccelCor.csv";header = true, delim = ",")
Collar8[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar8[4]))
Collar8[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar8[5]))
Collar8[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar8[6]))

#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar8[7])-100
    Collar8[7][i:i+2] .= mean(Collar8[7][i+3:i+99])
    Collar8[8][i:i+2] .= mean(Collar8[8][i+3:i+99])
    Collar8[9][i:i+2] .= mean(Collar8[9][i+3:i+99])
end


Collar8[:TotalG] = map((x,y,z) -> (x + y + z),Collar8[4],Collar8[5],Collar8[6])
Collar8[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar8[4],Collar8[5],Collar8[6])
Classed8 =  RunLSTM(Collar8,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed8[2])

Collar9 = CSV.read("Collar9AccelCor.csv";header = true, delim = ",")
Collar9[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar9[4]))
Collar9[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar9[5]))
Collar9[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar9[6]))

#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar9[7])-100
    Collar9[7][i:i+2] .= mean(Collar9[7][i+3:i+99])
    Collar9[8][i:i+2] .= mean(Collar9[8][i+3:i+99])
    Collar9[9][i:i+2] .= mean(Collar9[9][i+3:i+99])
end


Collar9[:TotalG] = map((x,y,z) -> (x + y + z),Collar9[4],Collar9[5],Collar9[6])
Collar9[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar9[4],Collar9[5],Collar9[6])
Classed9 =  RunLSTM(Collar9,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed9[2])

Collar12 = CSV.read("Collar12AccelCor.csv";header = true, delim = ",")
Collar12[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar12[4]))
Collar12[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar12[5]))
Collar12[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar12[6]))

#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar12[7])-100
    Collar12[7][i:i+2] .= mean(Collar12[7][i+3:i+99])
    Collar12[8][i:i+2] .= mean(Collar12[8][i+3:i+99])
    Collar12[9][i:i+2] .= mean(Collar12[9][i+3:i+99])
end


Collar12[:TotalG] = map((x,y,z) -> (x + y + z),Collar12[4],Collar12[5],Collar12[6])
Collar12[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar12[4],Collar12[5],Collar12[6])
Classed12 =  RunLSTM(Collar12,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed12[2])

Collar13 = CSV.read("Collar13AccelCor.csv";header = true, delim = ",")
Collar13[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar13[4]))
Collar13[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar13[5]))
Collar13[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar13[6]))

#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar13[7])-100
    Collar13[7][i:i+2] .= mean(Collar13[7][i+3:i+99])
    Collar13[8][i:i+2] .= mean(Collar13[8][i+3:i+99])
    Collar13[9][i:i+2] .= mean(Collar13[9][i+3:i+99])
end


Collar13[:TotalG] = map((x,y,z) -> (x + y + z),Collar13[4],Collar13[5],Collar13[6])
Collar13[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar13[4],Collar13[5],Collar13[6])
Classed13 =  RunLSTM(Collar13,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed13[2])


Collar14 = CSV.read("Collar14AccelCor.csv";header = true, delim = ",")
Collar14[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar14[4]))
Collar14[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar14[5]))
Collar14[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar14[6]))

#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar14[7])-100
    Collar14[7][i:i+2] .= mean(Collar14[7][i+3:i+99])
    Collar14[8][i:i+2] .= mean(Collar14[8][i+3:i+99])
    Collar14[9][i:i+2] .= mean(Collar14[9][i+3:i+99])
end


Collar14[:TotalG] = map((x,y,z) -> (x + y + z),Collar14[4],Collar14[5],Collar14[6])
Collar14[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar14[4],Collar14[5],Collar14[6])
Classed14 =  RunLSTM(Collar14,100,[4,5,6,7,8,9,10,11],6,5,TrainedModel14)
histogram(Classed14[2])

CSV.write("ClassedCollarData1.csv",Classed1)
CSV.write("ClassedCollarData2.csv",Classed2)
CSV.write("ClassedCollarData3.csv",Classed3)
CSV.write("ClassedCollarData4.csv",Classed4)
CSV.write("ClassedCollarData5.csv",Classed5)
CSV.write("ClassedCollarData6.csv",Classed6)
CSV.write("ClassedCollarData7.csv",Classed7)
CSV.write("ClassedCollarData8.csv",Classed8)
CSV.write("ClassedCollarData9.csv",Classed9)
CSV.write("ClassedCollarData10.csv",Classed10)
CSV.write("ClassedCollarData12.csv",Classed12)
CSV.write("ClassedCollarData13.csv",Classed13)
CSV.write("ClassedCollarData14.csv",Classed14)

##### wet season
cd("$(homedir())/Desktop/Work/Collars")
Collar1Wet = CSV.read("collar #1/ACCLOG00.csv";header = true, delim = ",")
Collar1Wet[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar1Wet[3]))
Collar1Wet[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar1Wet[4]))
Collar1Wet[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar1Wet[5]))


#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar1Wet[7])-100
    Collar1Wet[6][i:i+2] .= mean(Collar1Wet[6][i+3:i+99])
    Collar1Wet[7][i:i+2] .= mean(Collar1Wet[7][i+3:i+99])
    Collar1Wet[8][i:i+2] .= mean(Collar1Wet[8][i+3:i+99])
end


Collar1Wet[:TotalG] = map((x,y,z) -> (x + y + z),Collar1Wet[3],Collar1Wet[4],Collar1Wet[5])
Collar1Wet[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar1Wet[3],Collar1Wet[4],Collar1Wet[5])
Classed1Wet =  RunLSTM(Collar1Wet,100,[3,4,5,6,7,8,9,10],6,5,TrainedModel14)

Collar2Wet = CSV.read("collar #2/ACCLOG00.csv";header = true, delim = ",")
Collar2Wet[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar2Wet[3]))
Collar2Wet[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar2Wet[4]))
Collar2Wet[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar2Wet[5]))


#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar2Wet[6])-100
    Collar2Wet[6][i:i+2] .= mean(Collar2Wet[6][i+3:i+99])
    Collar2Wet[7][i:i+2] .= mean(Collar2Wet[7][i+3:i+99])
    Collar2Wet[8][i:i+2] .= mean(Collar2Wet[8][i+3:i+99])
end


Collar2Wet[:TotalG] = map((x,y,z) -> (x + y + z),Collar2Wet[3],Collar2Wet[4],Collar2Wet[5])
Collar2Wet[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar2Wet[3],Collar2Wet[4],Collar2Wet[5])
Classed2Wet =  RunLSTM(Collar2Wet,100,[3,4,5,6,7,8,9,10],6,5,TrainedModel14)

Collar5Wet = CSV.read("collar #5/ACCLOG00.csv";header = true, delim = ",")
Collar5Wet[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar5Wet[3]))
Collar5Wet[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar5Wet[4]))
Collar5Wet[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar5Wet[5]))


#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar5Wet[7])-100
    Collar5Wet[6][i:i+2] .= mean(Collar5Wet[6][i+3:i+99])
    Collar5Wet[7][i:i+2] .= mean(Collar5Wet[7][i+3:i+99])
    Collar5Wet[8][i:i+2] .= mean(Collar5Wet[8][i+3:i+99])
end


Collar5Wet[:TotalG] = map((x,y,z) -> (x + y + z),Collar5Wet[3],Collar5Wet[4],Collar5Wet[5])
Collar5Wet[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar5Wet[3],Collar5Wet[4],Collar5Wet[5])
Classed5Wet =  RunLSTM(Collar5Wet,100,[3,4,5,6,7,8,9,10],6,5,TrainedModel14)

Collar9Wet = CSV.read("collar #9/ACCLOG00.csv";header = true, delim = ",")
Collar9Wet[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar9Wet[3]))
Collar9Wet[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar9Wet[4]))
Collar9Wet[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar9Wet[5]))


#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar9Wet[7])-100
    Collar9Wet[6][i:i+2] .= mean(Collar9Wet[6][i+3:i+99])
    Collar9Wet[7][i:i+2] .= mean(Collar9Wet[7][i+3:i+99])
    Collar9Wet[8][i:i+2] .= mean(Collar9Wet[8][i+3:i+99])
end


Collar9Wet[:TotalG] = map((x,y,z) -> (x + y + z),Collar9Wet[3],Collar9Wet[4],Collar9Wet[5])
Collar9Wet[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar9Wet[3],Collar9Wet[4],Collar9Wet[5])
Classed9Wet =  RunLSTM(Collar9Wet,100,[3,4,5,6,7,8,9,10],6,5,TrainedModel14)

Collar11Wet = CSV.read("collar #11/ACCLOG00.csv";header = true, delim = ",")
Collar11Wet[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar11Wet[3]))
Collar11Wet[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar11Wet[4]))
Collar11Wet[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar11Wet[5]))


#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar11Wet[7])-100
    Collar11Wet[6][i:i+2] .= mean(Collar11Wet[6][i+3:i+99])
    Collar11Wet[7][i:i+2] .= mean(Collar11Wet[7][i+3:i+99])
    Collar11Wet[8][i:i+2] .= mean(Collar11Wet[8][i+3:i+99])
end


Collar11Wet[:TotalG] = map((x,y,z) -> (x + y + z),Collar11Wet[3],Collar11Wet[4],Collar11Wet[5])
Collar11Wet[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar11Wet[3],Collar11Wet[4],Collar11Wet[5])
Classed11Wet =  RunLSTM(Collar11Wet,100,[3,4,5,6,7,8,9,10],6,5,TrainedModel14)

Collar12Wet = CSV.read("collar #12/ACCLOG00.csv";header = true, delim = ",")
Collar12Wet[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar12Wet[3]))
Collar12Wet[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar12Wet[4]))
Collar12Wet[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar12Wet[5]))


#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar12Wet[7])-100
    Collar12Wet[6][i:i+2] .= mean(Collar12Wet[6][i+3:i+99])
    Collar12Wet[7][i:i+2] .= mean(Collar12Wet[7][i+3:i+99])
    Collar12Wet[8][i:i+2] .= mean(Collar12Wet[8][i+3:i+99])
end


Collar12Wet[:TotalG] = map((x,y,z) -> (x + y + z),Collar12Wet[3],Collar12Wet[4],Collar12Wet[5])
Collar12Wet[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar12Wet[3],Collar12Wet[4],Collar12Wet[5])
Classed12Wet =  RunLSTM(Collar12Wet,100,[3,4,5,6,7,8,9,10],6,5,TrainedModel14)

Collar14Wet = CSV.read("collar #14/ACCLOG00.csv";header = true, delim = ",")
Collar14Wet[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar14Wet[3]))
Collar14Wet[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar14Wet[4]))
Collar14Wet[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar14Wet[5]))


#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar14Wet[7])-100
    Collar14Wet[6][i:i+2] .= mean(Collar14Wet[6][i+3:i+99])
    Collar14Wet[7][i:i+2] .= mean(Collar14Wet[7][i+3:i+99])
    Collar14Wet[8][i:i+2] .= mean(Collar14Wet[8][i+3:i+99])
end


Collar14Wet[:TotalG] = map((x,y,z) -> (x + y + z),Collar14Wet[3],Collar14Wet[4],Collar14Wet[5])
Collar14Wet[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar14Wet[3],Collar14Wet[4],Collar14Wet[5])
Classed14Wet =  RunLSTM(Collar14Wet,100,[3,4,5,6,7,8,9,10],6,5,TrainedModel14)

Collar16Wet = CSV.read("collar #16/ACCLOG00.csv";header = true, delim = ",")
Collar16Wet[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar16Wet[3]))
Collar16Wet[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar16Wet[4]))
Collar16Wet[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar16Wet[5]))


#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar16Wet[7])-100
    Collar16Wet[6][i:i+2] .= mean(Collar16Wet[6][i+3:i+99])
    Collar16Wet[7][i:i+2] .= mean(Collar16Wet[7][i+3:i+99])
    Collar16Wet[8][i:i+2] .= mean(Collar16Wet[8][i+3:i+99])
end


Collar16Wet[:TotalG] = map((x,y,z) -> (x + y + z),Collar16Wet[3],Collar16Wet[4],Collar16Wet[5])
Collar16Wet[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar16Wet[3],Collar16Wet[4],Collar16Wet[5])
Classed16Wet =  RunLSTM(Collar16Wet,100,[3,4,5,6,7,8,9,10],6,5,TrainedModel14)

Collar17Wet = CSV.read("collar #17/ACCLOG00.csv";header = true, delim = ",")
Collar17Wet[:Filtx] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar17Wet[3]))
Collar17Wet[:Filty] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar17Wet[4]))
Collar17Wet[:Filtz] = filt(digitalfilter(responsetype, designmethod),convert(Array{Float64},Collar17Wet[5]))


#filter still has a 1 point delay, replace the first point with mean of seq
for i = 1:100:length(Collar17Wet[7])-100
    Collar17Wet[6][i:i+2] .= mean(Collar17Wet[6][i+3:i+99])
    Collar17Wet[7][i:i+2] .= mean(Collar17Wet[7][i+3:i+99])
    Collar17Wet[8][i:i+2] .= mean(Collar17Wet[8][i+3:i+99])
end


Collar17Wet[:TotalG] = map((x,y,z) -> (x + y + z),Collar17Wet[3],Collar17Wet[4],Collar17Wet[5])
Collar17Wet[:AbsTotalG] = map((x,y,z) -> (abs(x) + abs(y) + abs(z) - 9.8),Collar17Wet[3],Collar17Wet[4],Collar17Wet[5])
Classed17Wet =  RunLSTM(Collar17Wet,100,[3,4,5,6,7,8,9,10],6,5,TrainedModel14)
