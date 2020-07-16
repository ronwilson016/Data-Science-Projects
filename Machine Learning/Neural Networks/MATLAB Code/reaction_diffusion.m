clear all; close all; clc

% lambda-omega reaction-diffusion system
%  u_t = lam(A) u - ome(A) v + d1*(u_xx + u_yy) = 0
%  v_t = ome(A) u + lam(A) v + d2*(v_xx + v_yy) = 0
%
%  A^2 = u^2 + v^2 and
%  lam(A) = 1 - A^2
%  ome(A) = -beta*A^2

t = 0:0.05:10;
d1 = 0.1; d2 = 0.1; beta = 1.0;
L = 20; n = 512; N = n*n;
x2 = linspace(-L/2,L/2,n+1); x = x2(1:n); y = x;
kx = (2*pi/L)*[0:(n/2-1) -n/2:-1]; ky = kx;

% INITIAL CONDITIONS
[X,Y] = meshgrid(x,y);
[KX,KY] = meshgrid(kx,ky);
K2 = KX.^2+KY.^2; K22 = reshape(K2,N,1);

m = 1; % number of spirals

u = zeros(length(x),length(y),length(t));
v = zeros(length(x),length(y),length(t));

u(:,:,1) = tanh(sqrt(X.^2+Y.^2)).*cos(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));
v(:,:,1) = tanh(sqrt(X.^2+Y.^2)).*sin(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));

%% REACTION-DIFFUSION
uvt = [reshape(fft2(u(:,:,1)),1,N) reshape(fft2(v(:,:,1)),1,N)].';
[t,uvsol] = ode45('reaction_diffusion_rhs',t,uvt,[],K22,d1,d2,beta,n,N);

Udata(:,1) = reshape(u(:,:,1),N,1); Vdata(:,1) = reshape(v(:,:,1),N,1);
for j = 1:length(t)-1
    ut = reshape((uvsol(j,1:N).'),n,n);
    vt = reshape((uvsol(j,(N+1):(2*N)).'),n,n);
    u(:,:,j+1) = real(ifft2(ut));
    v(:,:,j+1) = real(ifft2(vt));
    Udata(:,j+1) = reshape(u(:,:,j+1),N,1);
    Vdata(:,j+1) = reshape(v(:,:,j+1),N,1);
    
    figure(1)
    pcolor(x,y,v(:,:,j+1)); shading interp; colormap(hot); colorbar; drawnow;
end

%% SVD Analysis
data0 = [Udata; Vdata]; % collecting the data into one matrix
[u0,s0,v0] = svd(data0,'econ'); % reduced svd on data matrix
sig0 = diag(s0); var0 = sig0/sum(sig0); % computing the variance

% plotting singular values to determine rank r
figure(2), plot(var0*100,'ro','Linewidth',[2]), xlabel('Singular Values')
ylabel('% of Variance'), title('Singular Value Spectrum')

r = 4; % rank of the feature space
data = u0(:,1:r).'*data0; % low-rank data approximation

%% Training NN
numTimeStepsTrain = floor(0.9*length(t)); % length of training set
numTimeStepsTest = length(t)-(numTimeStepsTrain+1); % length of testing set
dataTrain = data(:,1:numTimeStepsTrain+1); % training data
dataTest = data(:,numTimeStepsTrain+1:end); % testing data

t0 = randi([1 numTimeStepsTest],1,1); % getting a random index for testing
test = dataTest(:,t0); % original image for testing

% getting parameters for standardizing the data
mu = zeros(length(r)); sig = zeros(length(r));
for j = 1:r
    mu(j) = mean(dataTrain(j,:));
    sig(j) = std(dataTrain(j,:));
end

dataTrainStandardized = (dataTrain - mu.') ./ sig.'; % standardizing training set
Xtrain = dataTrainStandardized(:,1:end-1); % input training set
Ytrain = dataTrainStandardized(:,2:end); % output training set

dataTestStandardized = (dataTest - mu.') ./ sig.'; % standardizing testing set
Xtest = dataTestStandardized(:,1:end-1); % input testing set

numFeatures = r; % size of input data
numResponses = r; % size of output data
numHiddenUnits = 200;
layers = [ ... % defining the layers of the NN
    sequenceInputLayer(numFeatures) % sequence input layer of size r
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
pre_traj = [dataTrain(:,numTimeStepsTrain) YPred]; % collecting predicted trajectories from NN
pre_test = pre_traj(:,t0); % getting prediction of the test image

% projecting testing image back into full domain
test_proj = u0(:,1:r)*test; % true trajectory
pre_proj = u0(:,1:r)*pre_test; % predicted trajectory

% plotting trajectory test
figure(3)
subplot(1,2,1)
pcolor(x,y,reshape(test_proj(1:length(test_proj)/2),n,n)), shading interp
colormap(hot), colorbar, title('True Trajectory')
subplot(1,2,2)
pcolor(x,y,reshape(pre_proj(1:length(pre_proj)/2),n,n)), shading interp
colormap(hot), colorbar, title('NN Trajectory')

Ytest = dataTest(:,2:end); % output testing set
rmse = sqrt(mean((YPred-Ytest).^2)); % calculating the rms error

% plotting the error of the NN
figure(4)
stem((YPred - Ytest).')
xlabel("Time"), ylabel("Error"), title("RMSE (Reaction-Diffusion)")