function [lossGrad lossGauss posGrad posGauss biasGrad biasGauss ...
    stepsGrad stepsGauss finalGrad finalGauss HXinf] ...
    = solveGPS(y0, SL, s0, b0, maxSteps, maxErr, aGrad, aGauss, sTrue, b)
            
% y0 is the pseudorange vector
% SL is the satellite positions matrix formatted as column <-> satellite
% s0 is the initial guess for S
% b0 is the initial guess for bias
% aGrad and aGauss are the step size coefficients
% sTrue is the actual receiver position, b is the actual bias
% HXinf is the Jacobian for error prediction
reclen = 5e3; %max data points to record for plotting
posGrad = zeros(reclen,1);
biasGrad = posGrad;
lossGrad = posGrad;

% Gradient Descent
xk = [s0; b0]; %initial condition vector
stepsGrad = 0;
y = y0;
loss = maxErr + 1;
while stepsGrad < maxSteps && loss > maxErr
    stepsGrad = stepsGrad + 1;
    Sk = xk(1:3); %position
    bk = xk(4); %bias
    for i = 1:length(SL(1,:))
        deltaS = Sk - SL(:,i);
        R = norm(deltaS);%the true range, as calculated from current estimate of position
        r(i,:) = deltaS' / R; %the unit vector of deltaS
        h(i,1) = R + bk;
    end
    H = [r ones(length(r(:,1)),1)]; %Jacobian matrix
    xk = xk + aGrad * H' * (y - h); %next value of x
    loss = .5 * sum((y-h).^2);
    
    %don't record data beyond the end of the array, but allow iteration
    %to continue until termination conditions are satisfied
    if stepsGrad <= reclen 
        posGrad(stepsGrad) = norm(sTrue-Sk);
        biasGrad(stepsGrad) = abs(b-bk);
        lossGrad(stepsGrad) = loss;
    end                   
end
finalGrad = xk(1:3);

% Gauss
xk = [s0; b0]; %initial condition vector
stepsGauss = 0;
loss = maxErr + 1;
while stepsGauss < 100 && loss > maxErr
    stepsGauss = stepsGauss + 1;
    Sk = xk(1:3);
    bk = xk(4);
    for i = 1:length(SL(1,:))
        deltaS = Sk - SL(:,i);
        R = norm(deltaS); 
        r(i,:) = deltaS' / R; %the unit vectors
        h(i,1) = R + bk; % form h
    end
    H = [r ones(length(r(:,1)),1)]; %Jacobian matrix
    xk = xk + aGauss * H\(y - h); %next value of x
    posGauss(stepsGauss) = norm(sTrue-Sk);
    biasGauss(stepsGauss) = abs(b-bk);
    loss = .5 * norm(y-h)^2;
    lossGauss(stepsGauss) = loss;
end
HXinf = H;
finalGauss = xk(1:3);
end
