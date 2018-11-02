#### AcceML Neural Network Machine Learning For Accelerometer Data
using Pkg
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Random
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

#function to take labeled training data csv and produce
#the training data required for the LSTM
function CreateTraining(Data,TSlen,features,correct)
   TrainDat = Dict()
   counter = 1
   Classifiers = convert(Array,unique(Data[correct]))
   for i = 1:TSlen:nrow(Data)
      curdat = []
      for j = 1:length(features)
          push!(curdat,Data[i:i+(TSlen-1),features[j]])
      end
      curdat = reshape(hcat.(curdat),1,:,3)
      Class = convert(Array,Data[i,correct])
      TrainDat[counter] = [curdat,Class]
      counter +=1
   end
   for i = 1:length(TrainDat)
      Corary = zeros(1,length(Classifiers))
      Corary[findall(Classifiers .== TrainDat[i][2][1])[1][1]] = 1
      TrainDat[i][2] = Corary
   end
   return(TrainDat,Classifiers)
end


function BootstrapDat(TrainDat,minibatch,TSlen)
   KeyOrder = collect(1:length(TrainDat))
   KeyOrder = reshape(KeyOrder,minibatch,:)
   KeyOrder = transpose(KeyOrder)
   SampleOrder = KeyOrder[randperm(end),:]
   return(SampleOrder)
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
   counter = 1
   for i = diagind(y)
   y[i] = x[counter]
   counter += 1
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

function NLLDiff(x,y)
   z = []
   for i = 1:length(x)
      push!(z, y[i] * -1/x[i])
   end
   return sum(z)
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


##### time to build an lstm RNN

function initialiseLSTM(lengthinput,sizehiddenlayer,numberofclass,dims)
   We = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),(sizehiddenlayer * sizehiddenlayer)*dims),sizehiddenlayer,sizehiddenlayer,dims)
   Wf = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),(sizehiddenlayer * sizehiddenlayer)*dims),sizehiddenlayer,sizehiddenlayer,dims)
   Wg = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),(sizehiddenlayer * sizehiddenlayer)*dims),sizehiddenlayer,sizehiddenlayer,dims)
   Wq = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),(sizehiddenlayer * sizehiddenlayer)*dims),sizehiddenlayer,sizehiddenlayer,dims)
#
   be = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),dims,sizehiddenlayer),1,sizehiddenlayer,3)
   bf = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),dims,sizehiddenlayer),1,sizehiddenlayer,3)
   bg = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),dims,sizehiddenlayer),1,sizehiddenlayer,3)
   bq = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),dims,sizehiddenlayer),1,sizehiddenlayer,3)
#
   Ue = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),(sizehiddenlayer * lengthinput)*dims),sizehiddenlayer,lengthinput,dims)
   Uf = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),(sizehiddenlayer * lengthinput)*dims),sizehiddenlayer,lengthinput,dims)
   Ug = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),(sizehiddenlayer * lengthinput)*dims),sizehiddenlayer,lengthinput,dims)
   Uq = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),(sizehiddenlayer * lengthinput)*dims),sizehiddenlayer,lengthinput,dims)
#
   h_0 = reshape(zeros(1,sizehiddenlayer,dims),1,sizehiddenlayer,dims)
   s_0 = reshape(zeros(1,sizehiddenlayer,dims),1,sizehiddenlayer,dims)
#
   U = reshape(rand(Uniform(-1/sqrt(sizehiddenlayer),1/sqrt(sizehiddenlayer)),(numberofclass * sizehiddenlayer)*dims),numberofclass,sizehiddenlayer,dims)
   b2 = reshape(rand(Uniform(-1/sqrt(numberofclass),1/sqrt(numberofclass)),dims,numberofclass),1,numberofclass,3)
#
   return    params = Dict([("We",We),("Wf",Wf),("Wg",Wg),("Wq",Wq),
   ("be",be),("bf",bf),("bg",bg),("bq",bq),
   ("Ue",Ue),("Uf",Uf),("Ug",Ug),("Uq",Uq),("h_0", h_0),("s_0",s_0),("U",U),("b2",b2)])
end

function LSTMForward(x_t,h_prev,s_prev,Params)
   dims = size(x_t)[3]
   # compute gate values
   f_t = []
   e_t = []
   g_t = []
   q_t = []
   #
   s_next = []
   h_next = []
   for i = 1:dims
      f_t = push!(f_t,Sigmoid.(Params["bf"][:,:,i] .+ (transpose(x_t[i]) * transpose(Params["Uf"][:,:,i])) + (h_prev[:,:,i] * transpose(Params["Wf"][:,:,i]))))
      e_t = push!(e_t,tanh.(Params["be"][:,:,i] .+ (transpose(x_t[i]) * transpose(Params["Ue"][:,:,i])) + (h_prev[:,:,i] * transpose(Params["We"][:,:,i]))))
      g_t = push!(g_t,Sigmoid.(Params["bg"][:,:,i] .+ (transpose(x_t[i]) * transpose(Params["Ug"][:,:,i])) + (h_prev[:,:,i] * transpose(Params["Wg"][:,:,i]))))
      q_t = push!(q_t,Sigmoid.(Params["bq"][:,:,i] .+ (transpose(x_t[i]) * transpose(Params["Uq"][:,:,i])) + (h_prev[:,:,i] * transpose(Params["Wq"][:,:,i]))))
      #compute signals
      s_next = push!(s_next,dot.(f_t[i],s_prev[i]) + dot.(g_t[i],e_t[i]))
      h_next = push!(h_next,dot.(q_t[i], tanh.(s_next[i])))
   end
   s_next = reshape(s_next,1,:,dims)
   h_next = reshape(h_next,1,:,dims)
   cache = Dict([("s_prev",s_prev),("s_next",s_next),("x_t",x_t),
   ("e_t",e_t),("f_t", f_t), ("g_t",g_t),("q_t",q_t),("h_prev",h_prev)])
   return h_next, s_next, cache
end

function LSTMForwardPass(x,Params)
   a,b,d = size(Params["h_0"])
   c = size(x)[1]
   h = zeros(a,b,d,c)
   h_prev = Params["h_0"]
   s_prev = Params["s_0"]
   cache_dict = Dict()
   for i = 1:c
      h_temp, s_next,cache_step = LSTMForward(x[i],h_prev,s_prev,Params)
      for j in 1:d
         h[:,:,j,i] = h_temp[j]
      end
      h_prev = h[:,:,:,i]
      s_prev = s_next
      cache_dict[i] = cache_step
   end
   return h,cache_dict
end

function LSTMBackwards(dh_next,ds_next,Cache,Params)
   tanh_s = tanh.(cache["s_next"])
   ds_next = dot.((dot.(dh_next, Cache["q_t"])),TanhDiff.(Cache["s_next"])) + ds_next
   #forget gate f
   df_step = dot.(ds_next, Cache["s_prev"])
   dsigmoid_f = SigmoidDiff.(Cache["f_t"])
   f_temp = dot.(df_step,dsigmoid_f)
   dUf_step = transpose(f_temp) * Cache["x_t"]
   dWf_step = transpose(f_temp) * Cache["h_prev"]
   dbf_step =  transpose(repeat([sum(f_temp)],size(f_temp)[2]))
   #Input gate g
   dg_step = dot.(ds_next, Cache["e_t"])
   dsigmoid_g = SigmoidDiff.(Cache["g_t"])
   g_temp = dot.(dg_step, dsigmoid_g)
   dUg_step = transpose(g_temp) * Cache["x_t"]
   dWg_step = transpose(g_temp) * Cache["h_prev"]
   dbg_step =  transpose(repeat([sum(g_temp)],size(g_temp)[2]))
   #output gate q
   dq_step = dot.(ds_next, tanh_s)
   dsigmoid_q = SigmoidDiff.(Cache["q_t"])
   q_temp = dot.(dq_step, dsigmoid_q)
   dUq_step = transpose(q_temp) * Cache["x_t"]
   dWq_step = transpose(q_temp) * Cache["h_prev"]
   dbq_step = transpose(repeat([sum(q_temp)],size(q_temp)[2]))
   #input transform e
   de_step = dot.(ds_next, Cache["g_t"])
   dsigmoid_e = SigmoidDiff.(Cache["e_t"])
   e_temp = dot.(de_step, dsigmoid_e)
   dUe_step = transpose(e_temp) * Cache["x_t"]
   dWe_step = transpose(e_temp) * Cache["h_prev"]
   dbe_step =  transpose(repeat([sum(e_temp)],size(e_temp)[2]))
   #gradient w.r.t previous state h_prev
   dh_prev = dot.(dh_next, dot.(tanh_s, dsigmoid_q))* Params["Wq"] +
   dot.(ds_next,dot.(s_prev,dsigmoid_f)) * Params["Wf"]+
   dot.(ds_next,dot.(g_t,dsigmoid_e)) * Params["We"] +
   dot.(ds_next,dot.(e_t,dsigmoid_g)) * Params["Wg"]
   ds_prev = dot.(Cache["f_t"],ds_next)
   grads = Dict([("We" , dWe_step), ("Wf" , dWf_step), ("Wg" , dWg_step), ("Wq", dWq_step),
              ("Ue" , dUe_step), ("Uf" , dUf_step), ("Ug" , dUg_step), ("Uq" , dUq_step),
              ("be" , dbe_step), ("bf" , dbf_step), ("bg" , dbg_step), ("bq" , dbq_step)])
   return dh_prev, ds_prev, grads
end

function LSTMBackwardProp(dh, cache_dict, Params)
   a,b,c = size(dh)
   dh_next = zeros(a,b)
   ds_next = zeros(a,b)
   all_grads = Dict()
   kys = collect(keys(Params))
   kys = filter!(e->e∉["s_0","h_0","U","b2"],kys)
   for (n, f) in enumerate(kys)
      a,b = size(Params[f])
      all_grads[f] = zeros(a,b)
   end
   for i = c:-1:2
      dh_next = dh[:,:,i-1]
      dh_prev, ds_prev, step_grads = LSTMBackwards(dh_next, ds_next, cache_dict[i-1], Params)
      dh_next = dh_prev
      ds_next = ds_prev
      for (n, k) in enumerate(kys)
         all_grads[k] = -(all_grads[k] + step_grads[k])
      end
   end
   return all_grads
end

function LSTMAfflineFW(h,U,b2)
   a,b,c,d = size(h)
   e = size(b2)[2]
   theta = zeros(a,e,c,d)
   y = zeros(a,e,c,d)
   ypred = []
   for i = 1:d
      for j = 1:c
         theta[:,:,j,i] = (h[:,:,j,i] * transpose(U[:,:,j])) + b2[:,:,j]
         y[:,:,j,i] = Softmax(theta[:,:,j,i])
      end
      for z = 1:e
         y3 = []
         for j = 1:c
         y3 = push!(y3,y[:,:,j,:][z])
         end
         ypred = push!(ypred,mean(y3))
      end
   end
   Cache = U,b2,h
   return theta,y, Cache,ypred
end

function LSTMAfflineBW(theta,y,yt,Cache)
   U,b2,h = Cache
   a,b,c = size(theta)
   sdth = zeros(b,b,c)
   loss = []
   Losdif= zeros(a,b,c)
   dthetah = zeros(a,b,c)
   dU = zeros(size(U)[1],size(U)[2],c)
   for i = 1:size(theta)[3]
      sdth[:,:,i] = SoftmaxDiff(theta[:,:,i])
      push!(loss,NLL(y[:,:,i],yt[:,:,i]))
      Losdif[:,:,i] = transpose(NLLDiff(y[:,:,i],yt[:,:,i]))
      dtheta[:,:,i] = Losdif[:,:,i] * sdth[:,:,i]
      dh[:,:,i] = dtheta[:,:,i] * U
      dU[:,:,i] =  transpose(dtheta[:,:,i]) * dh
   end
   db2 = repeat([-sum(dtheta)],size(dtheta)[2],1)
   return dtheta,dh,dU,db2,loss
end

function CreateDataArray(Order,dict)
   Data = []
   for i in Order
      push!(Data,dict[i][1])
   end
   return Data
end

function TrainLSTM(InputDat,TSlen,hiddim,batchlen,Numclas,Report = -1,features,iter)
   Params = initialiseLSTM(TSlen,hiddim,Numclas,features)
   counter = 1
   for i = 1:iter
      CurrentOrder = BootstrapDat(InputDat,batchlen,TSlen)
      Data = CreateDataArray(CurrentOrder,InputDat)
      for i = 1:length(Data)
         counter += 1
         Fwh,Fwcache = LSTMForwardPass(Data[1:batchlen],Params)
         Clas,the,Afcache = LSTMAfflineFW(Fwh,Params["U"],Params["b2"])
         dtheta,dh,dU,db2,loss = LSTMAfflineBW(Clas,the,Afcache)
         all_grads = LSTMBackwardProp(dh,Fwcache,Params)
         all_grads["U"] = -dU
         all_grads["b2"] = db2
         kys = collect(keys(all_grads))
         for (n, k) in enumerate(kys)
            Params[k] = (Params[k] + all_grads[k])
         end
         if counter == Report
            print(string("Loss is",loss))
            counter = 1
         end
      end
   end
   return Params
end

function RunLSTM(InputDat,hiddim,batchlen,Numclas)
   Params = initialiseLSTM(size(Inputdat)[1],hiddim,Numclas)
   classedDat = InputDat
   for i=1:length(InputDat):batchlen
      Fwh,Fwcache = LSTMForwardPass(InputDat[i:i+batchlen],Params)
      Clas,the,Afcache = LSTMAfflineFW(Fwh,Params["U"],Params["b2"])
      for j = i:batchlen
         push!(classedDat[j],argmax(Clas)[2])
      end
   end
   return classedDat
end
