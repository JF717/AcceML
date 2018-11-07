
function softmaxloss(theta,y)
    a,b,c,d = size(theta)
    loss = []
    combloss = []
    for i = 1:d
        for j = 1:c
            prbs = exp.(theta[:,:,j,i] .- maximum(theta[:,:,j,i]))
            prbs = prbs ./ sum(prbs)
            loss = push!(loss,-sum(log.(prbs[findall(yt[i] .== 1)])))
            tempdtheta = prbs
            tempdtheta[findall(yt[i] .== 1)] = tempdtheta[findall(yt[i] .== 1)] .- 1
            tempdtheta = tempdtheta ./ b
            dtheta[:,:,j,i] = tempdtheta
        end
    end
    for i = 1:c:c*d
        combloss = push!(combloss,mean(loss[i:i+2]))
    end
    return combloss, dtheta
end
