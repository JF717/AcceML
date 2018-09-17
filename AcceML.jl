#### AcceML Neural Network Machine Learning For Accelerometer Data
using Pkg
using CSV
using DataFrames
using Distributions
using LinearAlgebra

#Collar2 = CSV.read("Collar2AccelCor.csv"; header = true, delim = ",")
#Collar2 = Collar2[:,2:6]

# If your data is not in Neural Network format this will transpose data into input arrays of whatever size you want
#Input data must be in format Date Time X Y Z
#Becomes Date Time x1:xn y1:yn z1:1n n = number of input neurones
function TransposeAccel(Data, Windows, Hz)
 obj = Data
 TransposedWindows = Dict()
 for j = 1:length(Windows)
 curtim = Windows[j] * Hz
 Transobj = Any[]
 dat = []
 tim = []
 x = []
 y = []
 z = []
 test = obj[:,3]
 for i = range(1; stop =  (nrow(obj)), step = curtim)
   dat = obj[i,1]
   tim = string(obj[i,2],"-", obj[i+(curtim - 1),2])
   x =  obj[i:(i+(curtim -1)),3]
   y = obj[i:(i+(curtim -1)),4]
   z = obj[i:(i+(curtim -1)),5]
   tempdata = Any[]
   push!(tempdata,dat)
   push!(tempdata,tim)
   for i = 1:length(x)
      push!(tempdata,x[i])
   end
   for i = 1:length(y)
      push!(tempdata,y[i])
   end
   for i = 1:length(z)
      push!(tempdata,z[i])
   end
   push!(Transobj,tempdata)
   if (i + (2* curtim) - 1) > nrow(obj)
      break
   end
   end
   TransposedWindows[j] = string("Window",Windows[j]) => Transobj
   end
   return TransposedWindows
end

#Sigmoid function transforms data so it is either 0 or 1 used for binary classification
function Sigmoid(x)
   return 1.0 ./ (1.0 .+ exp(-x))
end

#Softmax transforms between 0 and 1 with weights of total, used for multi classification
function Softmax(x)
   exp.(x) ./ sum(exp.(x))
end

#ReLU transforms all negative values into 0. You lose gradient descent power
function ReLU(x)
   if x < 0
      x = 0
   end
   return x
end

#LeakyReLU transforms negative values into a slight negative with an alpha normally 0.01
function LeakyReLU(x,a = 0.01)
   if x < 0
      x = (x *  a)
   end
   return x
end


#function to create the initial weights arrays for the NN
function InitialiseNet(LayNeur,InitMethod)
   #LayNeur is the number of neurones you want in each
   #layer including the input data size and your output layer
   #doesn't need initial weights if it is sigmoid / softmax
   #InitMethod is the initialisation of weights.
   NumWeights = Dict()
   #The Number of weights is the number of neurones in 1
   #layer times the next, essentially the connections
   for i = 1:(length(LayNeur)-1)
      NumWeights[i] = string("NumWeights",i) => LayNeur[i] * LayNeur[i+1]
   end
   Weights = Dict()
   #Deciding what initialisation method to use depends on the
   #activation function, generally GU for uniform distributions
   #GN for normal distributions and He for ReLU / LeakyReLU
   for i = 1:(length(LayNeur)-1)
      if InitMethod[i] == "GU"
         a = sqrt(6/(LayNeur[i] + LayNeur[i+1]))
         d = Uniform(-a, a)
         r = rand(d, NumWeights[i][2])
         r = reshape(r,LayNeur[i],LayNeur[i+1])
         Weights[i] = string("WeightsLayer",i) => r
      end
      if InitMethod[i] == "He"
         Var = sqrt(2/LayNeur[i])
         d = Normal(0.0,Var)
         r = rand(d, NumWeights[i][2])
         r = reshape(r,LayNeur[i],LayNeur[i+1])
         Weights[i] = string("WeightsLayer",i) => r
      end
      if InitMethod[i] == "GN"
         Var = sqrt(2/(LayNeur[i]+ LayNeur[i+1]))
         d = Normal(0.0,Var)
         r = rand(d, NumWeights[i][2])
         r = reshape(r,LayNeur[i],LayNeur[i+1])
         Weights[i] = string("WeightsLayer",i) => r
      end
   end
   return Weights
end

#function to do matrix dot multiplication like numpy.
function dotprodmat(x,y)
   out = []
   for i = range(1; stop = length(y), step = length(x))
      push!(out,dot(x,y[i:(i+length(x)-1)]))
   end
   return out
end

#function that runs the forward propagation of input data
#through the network. Provide it with the input data,
#the initial weights and the activation functions of each layer
function ForwardPass(Input,Weights,ActivFuns)
   z = Dict()
   a = Dict()
   z[1] = "Input" => Input
   a[1] = "Input" => Input
   cur = []
   for i = 1:length(Weights)
      layername = string("Layer",i)
      if i == length(Weights)
         layername = string("Output")
      end
      cur =  dotprodmat(z[i][2],Weights[i][2])
      a[i+1] = string(layername,"unact") => cur
      if ActivFuns[i] == "Sigmoid"
         z[i+1] = layername => Sigmoid.(cur)
      end
      if ActivFuns[i] == "Softmax"
         z[i+1] = layername => Softmax(cur)
      end
      if ActivFuns[i] == "tanh"
         z[i+1] = layername => tanh.(cur)
      end
      if ActivFuns[i] == "ReLU"
         z[i+1] = layername => ReLU.(cur)
      end
      if ActivFuns[i] == "LeakyReLU"
         z[i+1] = layername => LeakyReLU.(cur)
      end
   end
   return z,a
end

function SigmoidDiff(x)
   return Sigmoid(x) * (1-Sigmoid(x))
end

function TanhDiff(x)
   return sinh(x)/cosh(x)
end

function ReLUDiff(x)
   if x < 0
      x = 0
   else
      x = 1
   end
   return x
end

function LeakyReLUDiff(x,a = 0.01)
   if x < 0
      x = a
   else
      x = 1
   end
   return x
end

function SoftmaxDiff(x)
   y = zeros(length(x),length(x))
   for i = diagind(y)
   y[i] = x[Int(round(i/length(x)+0.1))]
   end
   for i = 1:size(y)[1]
      for j = 1:size(y)[1]
         print(i,j)
         if i == j
            y[i,j] = x[i] * (1-x[i])
         else
            y[i,j] = -x[i] * x[j]
         end
      end
   end
   return y
end

function MSE(x,y)
   z = []
   for i = 1:length(x)
      push!(z,(y[i] - x[i])^2)
   end
   return mean(z)
end

function MSEDiff(x,y)
   z = []
   for i = 1:length(x)
      push!(z,y[i]-x[i])
   end
   return z
end

function MSLE(x,y)
   z = []
   for i = 1:length(x)
      push!(z,log1p((y[i]) - log1p(x[i]))^2)
   end
   return mean(z)
end

function MSLEDiff(x,y)
   z = []
   for i = 1:length(x)
      push!(z, (log(y) - log(x)) / x)
   end
   return z
end

function L1(x,y)
   z = []
   for i = 1:length(x)
      push!(z,abs(y[i] - x[i]))
   end
   return sum(z)
end

function L1Diff(z,y)
   z = []
   for i = 1:length(x)
      push!(z,(y[i] - x[i])/(abs(y[i] - x[i])))
   end
   return z
end

function L2(x,y)
   z = []
   for i = 1:length(x)
      push!(z,(y[i] - x[i])^2)
   end
   return sum(z)
end

function L2Diff(x,y)
   z = []
   for i = 1:length(x)
      push!(z,2 * (y[i] - z[i]))
   end
   return z
end

function MAE(x,y)
   z = []
   for i = 1:length(x)
      push!(z,abs(y[i] - x[i]))
   end
   return (1/length(x)) * sum(z)
end

function MAEDiff(x,y)
   z = []
   for i = 1:length(x)
      if y[i] - x[i] > 0
      push!(z, 1)
      elseif y[i] - x[i] < 0
      push!(z, -1)
      else
      push!(z,0)
      end
   end
   return z
end

function MAPE(x,y)
   z = []
   for i = 1:length(x)
      push!(z,abs((y[i] - x[i]) / y[i])*100)
   end
   return (1/length(x)) * sum(z)
end

function Hinge(x,y,m = 1)
   z = []
   for i = 1:length(x)
      push!(z,max(0,m - y[i] * x[i]))
   end
   return (1/length(x)) * sum(z)
end

function HingeDiff(x,y,m=1)
   z = []
   for i = 1:length(x)
      if max(m,1 - y[i] * x[i]) < 1
         push!(z,-y[i]*x[i])
      elseif max(0,m - y[i] * x[i]) > 1
       push!(z,0)
      end
   end
   return z
end

function Hinge2(x,y,m = 1)
   z = []
   for i = 1:length(x)
      push!(z,max(0,m - y[i] * x[i])^2)
   end
   return (1/length(x)) * sum(z)
end


function NLL(x)
   for i = 1:length(x)
      if y[i] == 1
         return -log(x[i])
      end
   end
end

function NLLtot(x,y)
   push!(y,x)
   return sum(y)
end

function NLLDiff(x,y)
   z = []
   for i = 1:length(x)
      push!(z, -1/x[i])
   end
   return z
end

function xent(x,y)
   z = []
   for i = 1:length(x)
      push!(z,(y[i]*log(x[i])) + (1-y[i])*(log(1-x[i])))
   end
   return (-(1/length(x)) * sum(z))
end

function xentDiff(x,y)
   z = []
   for i = 1:length(x)
      push!(z,-1 * (1/x[i]) + (1-y[i])*(1/(1-x[i])))
   end
   return z
end

function MCXent(x,y)
   z = []
   for i = 1:length(x)
      push!(z,y[i]* log(x[i]))
   end
   return sum(z)
end

function MCXentDiff(x,y)
   z = []
   for i = 1:length(x)
      push!(z, 1/x[i])
   end
   return z
end

function KLdiv(x,y)
   z = []
   a = []
   for i = 1:length(x)
      push!(z,y[i] * log(y[i]))
      push!(a,y[i] * log(x[i]))
   end
   return ((1/length(x)) * sum(z)) - ((1/length(x))*sum(a))
end

function poissonloss(x,y)
   z = []
   for i = 1:length(x)
      push!(z, x[i] - (y[i]*log(x[i])))
   end
   return (1/length(x)) * sum(z)
end

function poissonDiff(x,y)
   z = []
   for i = 1:length(x)
      push!(z, 1 - 1 * 1/x[i])
   end
   return z
end


function cosprox(x,y)
   a = []
   b = []
   c = []
   for i = 1:length(x)
      push!(a,y[i]*x[i])
      push!(b,y[i]^2)
      push!(c,x[i]^2)
   end
   return sum(a) / sqrt(sum(b)) * sqrt(sum(c))
end

function cosproxDiff(x,y)
   z = []
   for i = 1:length(x)
      push!(z,x[i]/ (abs(y[i]) * abs(x[i]) - ((y[i] * x[i])/ (sqrt((y[i]^2)) * sqrt((x[i]^2))) * (y[i]/(abs(y[i])^2)))))
   end
   return z
end

function invdot(x,y)
   z = []
   for j = 1:length(x)
      for i = 1:length(y)
         push!(z,x[j] * y[i])
      end
   end
   return reshape(z,length(x),length(y))
end

function BackPropagation(x,y,lossfunc,ActivFuns,W,LR)
   z = Dict()
   # calculating the differential of the loss of ouput
   lf = getfield(Main,Symbol(string(lossfunc,"Diff")))
   Ed = lf(x[1][length(x[1])][2],y)
   #calculating the differential of the output activation
   af = getfield(Main,Symbol(string(ActivFuns[length(ActivFuns)],"Diff")))
   Od = af.(x[2][length(x[1])][2])
   EW = dot.(Ed,Od)
   z[length(W)] = "OutputLayerWeightGrad" => invdot(W[length(W)][2],EW)
   #these relative differentials are then used in every layer
   for i = (length(x[1])-1):-1:2
      #work out the layer error - the output of layer times the Error dif from
      #the first layer
      LE = invdot(x[1][i][2],EW)
      #get activation function and do diff of it on unactivated layer input
      af = getfield(Main,Symbol(string(ActivFuns[i],"Diff")))
      hd = af.(x[2][i][2])
      #dot product of this layers error and it's activation diff
      LEhd = dot.(LE,hd)
      #if we are on the final layer have to do it with the input instead
      if i == 2
         z[i-1] = string("Layer",(i-1),"WeightGrad") => invdot(x[1][1][2],LEhd)
      else
         #final gradient from the LEhd invdot with the previous layers weights
         z[i-1] = string("Layer",(i-1),"WeightGrad") => invdot(W[i-1][2],LEhd)
      end
   end
   NW = Dict()
   for i = 1:length(W)
      NW[i] = string("WeightsLayer",i) => W[i][2] - (LR * z[i][2])
   end
   return NW
end

function TrainANN(Input,Correct,Struc,InitMethod,ActivFuns,lossfunc,LR,Report)
   Wgts = InitialiseNet(Struc,InitMethod)
   lf = getfield(Main,Symbol(lossfunc))
   c = 1
   for i = 1:length(Input)
      F = ForwardPass(Input[i],Wgts,ActivFuns)
      if c == Report
         print(string("Loss:",lf(F[1][i][2],Correct[i])))
         c = 0
      end
      Wgts = BackPropagation(F,Correct[i],lossfunc,Wgts,LR)
      c += 1
   end
   print(string("New Synaptic weights after training:"))
   Wgts
   return Wgts
end

function RunANN(Input,Wgts,ActivFuns)
   clas = Dict()
   for i = 1:length(Input)
      O = ForwardPass(Input[i],Wgts,ActivFuns)
      clas[i] = Input[i] => O[1][i][2]
   end
   return clas
end

#initialising a simple RNN is a bit easier, we have a
#fixed number of layer weights but we still need to
#create the layers generally it will be a hidden
#and an output layer the size of our classification
#the initialisation function will relate to the
#activation function generally always sigmoid
#or tanh so we will use weights between -1 and 1
function InitialiseRNN(Lenin,LenHL,lenOC)
   #HL is hidden layer size
   #OC is output classification size
   z = Dict()
   z[1] = "IntoHid" => reshape(rand(Uniform(-1/sqrt(Lenin),1/sqrt(Lenin)),(Lenin * LenHL)),LenHL,Lenin)
   z[2] = "Hidwts" => reshape(rand(Uniform(-1/sqrt(LenHL),1/sqrt(LenHL)),(LenHL * LenHL)),LenHL,LenHL)
   z[3] = "Outwts" => reshape(rand(Uniform(-1/sqrt(LenHL),1/sqrt(LenHL)),(lenOC * LenHL)),LenHL,lenOC)
   return z
end

function RNNForward(Input,prev_o,W,ActivFun,dimin)
   if dimin < 2
      Input = reshape(Input,dimin,length(Input))
   end
   a = Input * transpose(W[1][2])
   b = prev_o * transpose(W[2][2])
   c = b + a
   af = getfield(Main,Symbol(ActivFun))
   next_o = af.(c)
   return prev_0, next_o
end

function RNNForwardPass(In,HL,W,ActivFun)
   prev_o = zeros(size(In)[1],HL)
   cache = Dict()
   for i = 1:length(In)
      prev_o,next_o, out = RNNForward(In[i],prev_o,W,ActivFun)
      cache[i] = prev_o,next_o, out
   end
   return cache
end

function RNNBackwards(Input,prev_o,next_o, W, ActivFun,Dout)
   afd = getfield(Main,Symbol(string(ActivFun,"Diff")))
   diff =  adf(next_o)
   intdp = dot.(diff,Dout)
   daf = getfield(Main,Symbol(string(ActivFun,"Diff")))
   odif = daf.(next_o)
   intdot = Dout * odif
   dW1_s = transpose(intdot)*prev_o
   dW2_s = transpose(intdot) * Input
   dW3_s = intdot * W
   return dW1, dW2, dW3
end

function RNNBackwardsprop(dh,cache,W)
   dW1 = zeros(size(W[1]))
   dW2 = zeros(size(W[2]))
   Dout = zeros(size(dh))
   for i = length(cache):1:-1
      Dout = dh[i-1]
      dW1_s,dW2_s,dW3_s = RNNBackwards(cache,cache,cache,W[],Dout)
      Dout = dW3_s
      dW1 +=dW1_s
      dW2 +=dW2_s
   end
   return dW1,dW2
end

function RNNAflineForward(out,W)
   th = out * transpose(W)
   y = Softmax.(th)
   return y, th
end

function RNNAflineBackwards(th,W,y,h)
   dth = SoftmaxDiff(th)
   loss = NLL(th,y)
   Losdif = NLLDiff(th,y)
   dtheta = dot.(dth,Losdif)
   dU = invdot(W,dtheta)
   dh = transpose(dtheta) * h
   return dtheta,dU,dh
end


##### time to build an lstm RNN

function initialiseLSTM(dimin,hiddim)
   We = reshape(rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),(hiddim * hiddim)),hiddim,hiddim)
   Wf = reshape(rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),(hiddim * hiddim)),hiddim,hiddim)
   Wg = reshape(rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),(hiddim * hiddim)),hiddim,hiddim)
   Wq = reshape(rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),(hiddim * hiddim)),hiddim,hiddim)
#
   be = rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),hiddim)
   bf = rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),hiddim)
   bg = rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),hiddim)
   bq = rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),hiddim)
#
   Ue = reshape(rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),(hiddim * hiddim)),hiddim,dimin)
   Uf= reshape(rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),(hiddim * hiddim)),hiddim,dimin)
   Ug= reshape(rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),(hiddim * hiddim)),hiddim,dimin)
   Uq= reshape(rand(Uniform(-1/sqrt(hiddim),1/sqrt(hiddim)),(hiddim * hiddim)),hiddim,dimin)
#
   return    params = Dict([("We",We),("Wf",Wf),("Wg",Wg),("Wq",Wq),
   ("be",be),("bf",bf),("bg",bg),("bq",bq),
   ("Ue",Ue),("Uf",Uf),("Ug",Ug),("Uq",Uq),])
end

function LSTMForward(x_t,h_prev,s_prev,Params)
   # compute gate values
   e_t = Sigmoid.(Params["be"] + (x_t * transpose(Params["Ue"]) + (h_prev * transpose(Params["We"])))
   f_t = Sigmoid.(Params["bf"] + (x_t * transpose(Params["Uf"]) + (h_prev * transpose(Params["Wf"])))
   g_t = Sigmoid.(Params["bg"] + (x_t * transpose(Params["Ug"]) + (h_prev * transpose(Params["Wg"])))
   q_t = Sigmoid.(Params["bq"] + (x_t * transpose(Params["Uq"]) + (h_prev * transpose(Params["Wq"])))
   #compute signals
   s_next = f_t * s_prev + g_t * e_t
   h_next = q_t * tanh.(s_next)
   cache = Dict([("s_prev",s_prev),("s_next",s_next),("x_t",x_t),
   ("e_t",e_t),("f_t", f_t), ("g_t",g_t),("q_t",q_t),("h_prev",h_prev)])
   return h_next, s_next, cache
end

function LSTMForwardPass(x,h_0,s_0,Params)
   h = Dict()
   h_prev = h_0
   s_prev = s_0
   cache_dict = Dict()
   for i = 1:length(x)
      h[i], s_next,cache_step = LSTMForward(x[i],h_prev,s_prev,Params)
      h_prev = h[i]
      s_prev = s_next
      cache_dict[i] = cache_step
   end
   return h,cache_dict
end

function LSTMBackwards()
end

function LSTMBackwardProp()
end

function LSTMAfflineFW()
end

function LSTMAfflineBW()
end
