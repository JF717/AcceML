#### AcceML Neural Network Machine Learning For Accelerometer Data
using Pkg
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Random

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
   #returns a dictionary with two parts one the data the other
   # is the correct one hot vector
   return(TrainDat,Classifiers)
end

#function to bootstrap the data so it is encountered
#in a different order in the next iteration
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

#differential of the Sigmoid
function SigmoidDiff(x)
   return Sigmoid(x) * (1-Sigmoid(x))
end

function TanhDiff(x)
   return sinh(x)/cosh(x)
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
         if i == j
            y[i,j] = x[i] * (1-x[i])
         else
            y[i,j] = -x[i] * x[j]
         end
      end
   end
   return y
end

function NLL(x,y)
   for i = 1:length(x)
      if y[i] == 1
         return -log(x[i])
      end
   end
end

function NLLDiff(x,y)
   z = []
   for i = 1:length(x)
      push!(z, y[i] * -1/x[i])
   end
   return sum(z)
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
      e_t = push!(e_t,Sigmoid.(Params["be"][:,:,i] .+ (transpose(x_t[i]) * transpose(Params["Ue"][:,:,i])) + (h_prev[:,:,i] * transpose(Params["We"][:,:,i]))))
      g_t = push!(g_t,Sigmoid.(Params["bg"][:,:,i] .+ (transpose(x_t[i]) * transpose(Params["Ug"][:,:,i])) + (h_prev[:,:,i] * transpose(Params["Wg"][:,:,i]))))
      q_t = push!(q_t,Sigmoid.(Params["bq"][:,:,i] .+ (transpose(x_t[i]) * transpose(Params["Uq"][:,:,i])) + (h_prev[:,:,i] * transpose(Params["Wq"][:,:,i]))))
      #compute signals
      s_next = push!(s_next,dot.(f_t[i], s_prev[i]) + dot.(g_t[i],e_t[i]))
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
   a,b,c = size(dh_next)
   d = size(Cache["x_t"][1])[1]
   tanh_s = zeros(a,b,c)
   dUf_step = zeros(b,d,c)
   dWf_step = zeros(b,b,c)
   dbf_step = zeros(a,b,c)
   #
   dUg_step = zeros(b,d,c)
   dWg_step = zeros(b,b,c)
   dbg_step = zeros(a,b,c)
   #
   dUq_step = zeros(b,d,c)
   dWq_step = zeros(b,b,c)
   dbq_step = zeros(a,b,c)
   #
   dUe_step = zeros(b,d,c)
   dWe_step = zeros(b,b,c)
   dbe_step = zeros(a,b,c)
   #
   dh_prev = zeros(a,b,c)
   ds_prev = zeros(a,b,c)
   for i in 1:size(dh_next)[3]
      #frequenctly used quantity
      tanh_s[:,:,i] = tanh.(Cache["s_next"][i])
      #internal state s
      ds_next[:,:,i] = dot.((dot.(dh_next[:,:,i], Cache["q_t"][i])),(1 .- TanhDiff.(Cache["s_next"][i]) .^ 2)) + ds_next[:,:,i]
      #forget gate f
      df_step = dot.(ds_next[:,:,i], Cache["s_prev"][i])
      dsigmoid_f = SigmoidDiff.(Cache["f_t"][i])
      f_temp = dot.(df_step,dsigmoid_f)
      dUf_step[:,:,i] = transpose(f_temp) * transpose(Cache["x_t"][i])
      dWf_step[:,:,i] = transpose(f_temp) * Cache["h_prev"][:,:,i]
      dbf_step[:,:,i] = repeat([sum(f_temp)],size(f_temp)[1],size(f_temp)[2])
      #Input gate g
      dg_step = dot.(ds_next[:,:,i], Cache["e_t"][i])
      dsigmoid_g = SigmoidDiff.(Cache["g_t"][i])
      g_temp = dot.(dg_step, dsigmoid_g)
      dUg_step[:,:,i] = transpose(g_temp) * transpose(Cache["x_t"][i])
      dWg_step[:,:,i] = transpose(g_temp) * Cache["h_prev"][:,:,i]
      dbg_step[:,:,i] = repeat([sum(g_temp)],size(g_temp)[1],size(g_temp)[2])
      #output gate q
      dq_step = dot.(ds_next[:,:,i], tanh_s[:,:,i])
      dsigmoid_q = SigmoidDiff.(Cache["q_t"][i])
      q_temp = dot.(dq_step, dsigmoid_q)
      dUq_step[:,:,i] = transpose(q_temp) * transpose(Cache["x_t"][i])
      dWq_step[:,:,i] = transpose(q_temp) * Cache["h_prev"][:,:,i]
      dbq_step[:,:,i] = repeat([sum(q_temp)],size(q_temp)[1],size(q_temp)[2])
      #input transform e
      de_step = dot.(ds_next[:,:,i], Cache["g_t"][i])
      dsigmoid_e = SigmoidDiff.(Cache["e_t"][i])
      e_temp = dot.(de_step, dsigmoid_e)
      dUe_step[:,:,i] = transpose(e_temp) * transpose(Cache["x_t"][i])
      dWe_step[:,:,i] = transpose(e_temp) * Cache["h_prev"][:,:,i]
      dbe_step[:,:,i] = repeat([sum(e_temp)],size(e_temp)[1],size(e_temp)[2])
      #gradient w.r.t previous state h_prev
      dh_prev[:,:,i] = dot.(dh_next[:,:,i], dot.(tanh_s[:,:,i], dsigmoid_q))* Params["Wq"][i] +
      dot.(ds_next[:,:,i],dot.(Cache["s_prev"][i],dsigmoid_f)) * Params["Wf"][i]+
      dot.(ds_next[:,:,i],dot.(Cache["g_t"][i],dsigmoid_e)) * Params["We"][i] +
      dot.(ds_next[:,:,i],dot.(Cache["e_t"][i],dsigmoid_g)) * Params["Wg"][i]
      ds_prev[:,:,i] = dot.(Cache["f_t"][i],ds_next[:,:,i])
   end
   grads = Dict([("We" , dWe_step), ("Wf" , dWf_step), ("Wg" , dWg_step), ("Wq", dWq_step),
               ("Ue" , dUe_step), ("Uf" , dUf_step), ("Ug" , dUg_step), ("Uq" , dUq_step),
               ("be" , dbe_step), ("bf" , dbf_step), ("bg" , dbg_step), ("bq" , dbq_step)])
   return dh_prev,ds_prev,grads
end

function LSTMBackwardProp(dh, cache_dict, Params)
   a,b,c,d = size(dh)
   dh_next = zeros(a,b,c)
   ds_next = zeros(a,b,c)
   all_grads = Dict()
   kys = collect(keys(Params))
   kys = filter!(e->e∉["s_0","h_0","U","b2"],kys)
   for (n, f) in enumerate(kys)
      a,b,c = size(Params[f])
      all_grads[f] = zeros(a,b,c)
   end
   for i = (d+1):-1:2
      dh_next = dh[:,:,:,i-1]
      dh_prev, ds_prev, step_grads = LSTMBackwards(dh_next, ds_next, cache_dict[i-1], Params)
      dh_next = dh_prev
      ds_next = ds_prev
      for (n, k) in enumerate(kys)
            all_grads[k] += step_grads[k]
      end
   end
   return all_grads
end

function LSTMAfflineFW(h,U,b2)
   a,b,c,d = size(h)
   e = size(b2)[2]
   theta = zeros(a,e,c,d)
   y = zeros(a,e,c,d)
   pred = zeros(a,e,d)
   for i = 1:d
      for j = 1:c
         theta[:,:,j,i] = (h[:,:,j,i] * transpose(U[:,:,j])) + b2[:,:,j]
         y[:,:,j,i] = Softmax(theta[:,:,j,i])
      end
   end
   for i = 1:d
      ypred = []
      for z = 1:e
         y3 = []
         for j = 1:c
            y3 = push!(y3,y[:,:,j,i][z])
         end
         ypred = push!(ypred,mean(y3))
      end
      pred[:,:,i] = ypred
   end
   Cache = U,b2,h
   return theta,y, Cache,pred
end
#this is wrong, need to do derivatives better
function LSTMAfflineBW(theta,y,yt,Cache)
   U,b2,h = Cache
   a,b,c,d = size(theta)
   sdth = zeros(b,b,c,d)
   loss = []
   meanloss = []
   Losdif= zeros(a,b,c,d)
   dtheta = zeros(a,b,c,d)
   dh = zeros(a,size(U)[2],c,d)
   dU = zeros(size(U)[1],size(U)[2],c,d)
   db2 = zeros(a,b,c)
   for i = 1:d
      for j = 1:c
         sdth[:,:,j,i] = SoftmaxDiff(theta[:,:,j,i])
      end
      for j = 1:c
         push!(loss,NLL(y[:,:,j,i],yt[i]))
      end
      push!(meanloss,mean(loss))
      for j = 1:c
         Losdif[:,:,j,i] = transpose(NLLDiff.(y[:,:,j,i],yt[i]))
         dtheta[:,:,j,i] = Losdif[:,:,j,i] * sdth[:,:,j,i]
         dh[:,:,j,i] = dtheta[:,:,j,i] * U[:,:,j]
         dU[:,:,j,i] =  transpose(dtheta[:,:,j,i]) * h[:,:,j,i]
      end
   end
   dU2 = zeros(size(U))
   for i = 1:d
      dU2 += dU[:,:,:,i]
   end
   for i = 1:c
      db2[:,:,i] = repeat([sum(dtheta[:,:,i,:])],a,b)
   end
   for dz in [dh,dU2,db2]
      clamp.(dz,-1,1)
   end
   return dtheta,dh,dU2,db2,loss
end

function CreateDataArray(Order,dict)
   Data = []
   for i in Order
      push!(Data,dict[i][1])
   end
   return Data
end

function CreateCorrectArray(Order,dict)
   Data = []
   for i in Order
      push!(Data,dict[i][2])
   end
   return Data
end

function RightWrong(pred,Correct,TP,TN,FP,FN)
   for i = 1:length(Correct)
      predc = findmax(pred[:,:,i])[2][2]
      truc = findmax(Correct[i])[2][2]
      if predc == truc
         TP += 1
         TN += (length(Correct)-1)
      else
         FP += 1
         FN += 1
         TN += (length(Correct)-2)
      end
   end
   return TP, TN, FP, FN
end


function TrainLSTM(InputDat,TSlen,hiddim,batchlen,Numclas,features,iter,LR = 1)
   Params = initialiseLSTM(TSlen,hiddim,Numclas,length(features))
   for i = 1:iter
      CurrentOrder = BootstrapDat(InputDat,batchlen,TSlen)
      Data = CreateDataArray(CurrentOrder,InputDat)
      Correct = CreateCorrectArray(CurrentOrder,InputDat)
      TP = 0
      TN = 0
      FP = 0
      FN = 0
      counter = 1
      for j = 1:batchlen:length(Data)
         counter += 1
         Fwh,Fwcache = LSTMForwardPass(Data[j:(j+batchlen-1)],Params)
         #print("\n",size(Fwh))
         the,Clas,Afcache,preds = LSTMAfflineFW(Fwh,Params["U"],Params["b2"])
         TP, TN, FP, FN = RightWrong(preds,Correct[j:(j+batchlen-1)],TP,TN,FP,FN)
         dtheta,dh,dU,db2,loss = LSTMAfflineBW(the,Clas,Correct[j:(j+batchlen-1)],Afcache)
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
         #print("\n",TP,"\n",TN,"\n",FP,"\n",FN)
         #print("\n",all_grads["We"][1])
         if counter == (length(Data)/batchlen)
            #print("\n",Params["We"][1])
            print("\n","Iteration ", i, " Accuracy is ",((TP+TN)/(TP +TN + FP +FN))*100,
            " Precision is ",((TP)/(TP + FP))*100,
            " Recall is ", (TP/(TP + FN)*100))
         end
      end
   end
   return Params
end

function RunLSTM(InputDat,hiddim,batchlen,Numclas,TrainedWeights)
   Params = TrainedWeights
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
