clear all; close all; clc

% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - u_xxxx,  periodic BCs 

N = 64; % N = 1024;
x = 32*pi*(1:N)'/N;
u = cos(x/16).*(1+sin(x/16)); 
v = fft(u);

% Spatial grid and initial condition:
h = 0.025;
k = [0:N/2-1 0 -N/2+1:-1]'/16;
L = k.^2 - k.^4;
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2)); 
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)); 
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop:
uu = u; tt = 0;
tmax = 100; nmax = round(tmax/h); nplt = floor((tmax/250)/h); g = -0.5i*k;
for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2);
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
    if mod(n,nplt)==0
        u = real(ifft(v));
        uu = [uu,u]; tt = [tt,t];
    end
end

%% Training NN
numTimeStepsTrain = floor(0.9*length(tt)); % length of training set
numTimeStepsTest = length(tt)-(numTimeStepsTrain+1); % length of testing set
dataTrain = uu(:,1:numTimeStepsTrain+1); % training data
dataTest = uu(:,numTimeStepsTrain+1:end); % testing data

% getting parameters for standardizing the data
mu = zeros(length(N)); sig = zeros(length(N));
for j = 1:N
    mu(j) = mean(dataTrain(j,:));
    sig(j) = std(dataTrain(j,:));
end

dataTrainStandardized = (dataTrain - mu.') ./ sig.'; % standardizing training set
Xtrain = dataTrainStandardized(:,1:end-1); % input training set
Ytrain = dataTrainStandardized(:,2:end); % output training set

dataTestStandardized = (dataTest - mu.') ./ sig.'; % standardizing testing set
Xtest = dataTestStandardized(:,1:end-1); % input testing set

numFeatures = N; % size of input data
numResponses = N; % size of output data
numHiddenUnits = 200;
layers = [ ... % defining the layers of the NN
    sequenceInputLayer(numFeatures) % sequence input layer of size N
    lstmLayer(numHiddenUnits) % LSTM layer with 200 hidden units
    fullyConnectedLayer(numResponses) % fully connected layer
    regressionLayer]; % regression output layer
options = trainingOptions('adam', ... % defining ADAM options for NN
    'MaxEpochs',100, ... % 100 iterations
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
net = trainNetwork(Xtrain,Ytrain,layers,options); % NN

%% preddicting responses from neural net to get evolution trajectories
net = predictAndUpdateState(net,Xtrain);
[net,YPred] = predictAndUpdateState(net,Ytrain(:,end));
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu'); % getting NN predictions
end
YPred = sig.'.*YPred + mu.'; % unstandardizing the predictions

Ytest = dataTest(:,2:end); % output testing set
rmse = sqrt(mean((YPred-Ytest).^2)); % calculating the rms error

pre_traj = [uu(:,numTimeStepsTrain) YPred]; % collecting predicted trajectories from NN
trajectories = [dataTrain(:,1:end-1) pre_traj]; % combining trajectories

%% visualizing results of NN
figure(1) % plotting the NN trajectories
plot(dataTrain(:,1:end-1).'), hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,pre_traj,'.-'), hold off
xlabel("Time"), ylabel("Trajectories"), title("Forecast (KS Equation)")

figure(2) % plotting the error
stem((YPred - Ytest).')
xlabel("Time"), ylabel("Error"), title("RMSE (KS Equation)")

%% setting different IC's for ode time-stepper
xt = 32*pi*(1:N)'/N;
ut = sin(xt/16).*(1+sin(xt/16)); % sin IC instead of cos
vt = fft(ut);

% solving the time-stepping loop with different IC's
uut = ut; ttt = 0;
for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(vt)).^2);
    a = E2.*vt + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2);
    b = E2.*vt + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    vt = E.*vt + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
    if mod(n,nplt)==0
        utt = real(ifft(vt));
        uut = [uut,utt]; ttt = [ttt,t];
    end
end

% plotting results
figure(3)
subplot(1,2,1), pcolor(xt,ttt,uut.'), shading interp
title('ODE Trajectories'), colormap(hot),axis off
subplot(1,2,2), pcolor(xt,ttt,trajectories.'), shading interp
title('NN Trajectories'), colormap(hot),axis off