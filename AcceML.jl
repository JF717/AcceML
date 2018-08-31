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
   return 1 / (1 + exp(-x))
end

#Softmax transforms between 0 and 1 with weights of total, used for multi classification
function Softmax(x)
   exp.(x) ./ sum(exp.(x))
end

#ReLU transforms all negative values into 0. You lose gradient descent power
function ReLU(x)
   if x[i] < 0
      x[i] = 0
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
   z[1] = "Input" => Input
   cur = []
   for i = 1:length(Weights)
      layername = string("Layer",i)
      if i == length(Weights)
         layername = string("Output")
      end
      cur =  dotprodmat(z[i][2],Weights[i][2])
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
   return z
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

function MSLE(x,y)
   z = []
   for i = 1:length(x)
      push!(z,log1p((y[i]) - log1p(x[i]))^2)
   end
   return mean(z)
end

function L1(x,y)
   z = []
   for i = 1:length(x)
      push!(z,abs(y[i] - x[i]))
   end
   return sum(z)
end

function L2(x,y)
   z = []
   for i = 1:length(x)
      push!(z,(y[i] - x[i])^2)
   end
   return sum(z)
end

function MAE(x,y)
   z = []
   for i = 1:length(x)
      push!(z,abs(y[i] - x[i]))
   end
   return (1/length(x)) * sum(z)
end

function MAPE(x,y)
   z = []
   for i = 1:length(x)
      push!(z,abs((y[i] - x[i]) / y[i])*100)
   end
   return (1/length(x)) * sum(z)
end

function Hinge(x,y,m = 0)
   z = []
   for i = 1:length(x)
      push!(z,max(0,1 - y[i] * x[i]))
   end
   return (1/length(x)) * sum(z)
end

function Hinge2(x,y,m = 0)
   z = []
   for i = 1:length(x)
      push!(z,max(0,1 - y[i] * x[i])^2)
   end
   return (1/length(x)) * sum(z)
end

function NLL(x,y)
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

function xent(x,y)
   z = []
   for i = 1:length(x)
      push!(z,(y[i]*log(x[i])) + (1-y[i])*(log(1-x[i])))
   end
   return (-1/length(x)) * sum(z)
end

function MCXent(x,y)
   z = []
   for i = 1:length(x)
      push!(z,y[i]* log(x[i]))
   end
   return sum(z)
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
