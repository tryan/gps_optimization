% Thomas Kok 

clc; clear;
% Raw data
ER2m = 6.37e6; % earth radius in meters
S = [1. 0. 0.]'; %true location
s0 = [.9331 .25 .258819]'; % initial guess for receiver location
Sat = [3.5852 2.9274 2.6612 1.4159; %satellite positions
       2.07 2.9274 0. 0.; 
       0. 0. 3.1712 3.8904];
b = 2.354788068e-3; % true clock bias
mcount = [1 4 16 256]; %number of measurements per satellite
stdevs = [.0004 .004]; %standard deviations for noise

y0 = zeros(length(Sat(1,:)),1);
for x = 1:length(Sat(1,:)) %get noiseless pseudorange values
    deltaS = Sat(:,x) - S;
    y0(x,1) = norm(deltaS) + b;
end

%noiseless data
[lossGrad lossGauss posGrad posGauss biasGrad biasGauss ...
    stepsGrad stepsGauss finalGrad finalGauss HXinf] ...
    = solveGPS(y0, Sat, s0, 0, 500000, 1e-15, .25, 1, S, b);

figure(1);
set(1,'color','w');
subplot(3,2,1);
plot(lossGrad);
title('Gradient Descent Noiseless');
ylabel('Loss Function');
subplot(3,2,3);
plot(posGrad*ER2m);
ylabel('Position Error (meters)');
xlabel(['Final Error: ' num2str(ER2m*norm(finalGrad-S)),'m',10,'Steps: ' num2str(stepsGrad)]);
subplot(3,2,5);
plot(biasGrad*ER2m);
ylabel('Bias Error (meters)');
xlabel('Iterations');

subplot(3,2,2);
plot(lossGauss);
title('Gaussian Descent Noiseless');
ylabel('Loss Function');
subplot(3,2,4);
plot(posGauss*ER2m);
ylabel('Position Error (meters)');
xlabel(['Final Error: ' num2str(ER2m*norm(finalGauss-S)),'m']);
subplot(3,2,6);
plot(biasGauss*ER2m);
ylabel('Bias Error (meters)');
xlabel('Iterations');
% pause;

% noisy data
for stdev = stdevs %add noise by averaging m numbers with mean = 0
    for m = mcount %and std = stdev and adding to y0
       for i = 1:length(y0) 
           y(i,1) = y0(i) + mean(normrnd(0,stdev,m,1));
       end
       
         [lossGrad lossGauss posGrad posGauss biasGrad biasGauss ...
       stepsGrad stepsGauss finalGrad finalGauss HXinf] ...
       = solveGPS(y, Sat, s0, 0, 500000, 1e-15, .25, 1, S, b);
   
       err = ER2m*norm(finalGauss-S);
       CovMat = stdev^2/m * (HXinf' * HXinf)^-1;
       exErr = sqrt(CovMat(1,1))*ER2m;

       subplot(3,2,1);
       plot(lossGrad);
       title(['Gradient Descent',10,'\sigma = ' num2str(stdev),10,'m = ' num2str(m)]);
       ylabel('Loss Function');
       subplot(3,2,3);
       plot(posGrad*ER2m);
       ylabel('Position Error (meters)');
       xlabel(['Final Error: ' num2str(ER2m*norm(finalGrad-S)),'m',10,'Steps: ' num2str(stepsGrad)]);
       subplot(3,2,5); 
       plot(biasGrad*ER2m);
       ylabel('Bias Error (meters)');
       xlabel('Iterations');
       
       subplot(3,2,2);
       plot(lossGauss);
       title(['Gaussian Descent',10,'\sigma = ' num2str(stdev),10,'m = ' num2str(m)]);
       ylabel('Loss Function');
       subplot(3,2,4);
       plot(posGauss*ER2m);
       ylabel('Position Error (meters)');
       xlabel(['Final Error: ' num2str(err),'m',10,'Expected Error: ',num2str(exErr),10,'Ratio: ',num2str(err/exErr)]);
       subplot(3,2,6);
       plot(biasGauss*ER2m);
       ylabel('Bias Error (meters)');
       xlabel('Iterations');
       pause;
    end
end


clear mcount stdevs deltaS S So x ER2m CovMat err exErr m stdev i stepsGrad stepsGauss y y0 Sat b s0 HXinf