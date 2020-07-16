clear all; close all; clc

% Simulate Lorenz system
dt = 0.01; T = 8; t = 0:dt:T;
b = 8/3; sig = 10;

% Rho = 10
r1 = 10;
Lorenz1 = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r1 * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

figure(1)
input1 = []; output1 = [];
for j = 1:100  % training trajectories
    x0 = 30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz1,t,x0);
    input1 = [input1; y(1:end-1,:)];
    output1 = [output1; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
grid on, view(-23,18), title('ODE Trajectories (rho = 10)')

% Rho = 28
r2 = 28;
Lorenz2 = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r2 * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

figure(2)
input2 = []; output2 = [];
for j = 1:100  % training trajectories
    x0 = 30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz2,t,x0);
    input2 = [input2; y(1:end-1,:)];
    output2 = [output2; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
grid on, view(-23,18), title('ODE Trajectories (rho = 28)')

% Rho = 40
r3 = 40;
Lorenz3 = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r3 * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

figure(3)
input3 = []; output3 = [];
for j = 1:100  % training trajectories
    x0 = 30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz3,t,x0);
    input3 = [input3; y(1:end-1,:)];
    output3 = [output3; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
grid on, view(-23,18), title('ODE Trajectories (rho = 40)')

% Training NN
n = randi([1 80000],1,0.9*80000); % generating indices for 90% training
input = [input1(n,:); input2(n,:); input3(n,:)]; % collecting the inputs
output = [output1(n,:); output2(n,:); output3(n,:)]; % collecting the outputs

net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.'); % manually stopped training at 60 epochs

%% Future-State Predictions for Rho = 17 & Rho = 35
r4 = 17; % rho = 17
Lorenz4 = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r4 * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

figure(4)
x0 = 20*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz4,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:) = x0;
for jj = 2:length(t)
    y0 = net(x0);
    ynn(jj,:) = y0.'; x0 = y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
legend('Trajectory','IC','Prediction'), title('NN Prediction (rho = 17)')

figure(5)
subplot(3,2,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2]), title('X direction (rho = 17)')
subplot(3,2,3), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2]), title('Y direction (rho = 17)')
subplot(3,2,5), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2]), title('Z direction (rho = 17)')

r5 = 35; % rho = 35
Lorenz5 = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r5 * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

figure(6)
x0 = 20*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz5,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:) = x0;
for jj = 2:length(t)
    y0 = net(x0);
    ynn(jj,:) = y0.'; x0 = y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
legend('Trajectory','IC','Prediction'), title('NN Prediction (rho = 35)')

figure(5)
subplot(3,2,2), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2]), title('X direction (rho = 35)')
subplot(3,2,4), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2]), title('Y direction (rho = 35)')
subplot(3,2,6), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2]), title('Z direction (rho = 35)')

figure(4), view(-75,15)
figure(6), view(-75,15)
figure(5)
subplot(3,2,1), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,2), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,3), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,4), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,5), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,6), set(gca,'Fontsize',[15],'Xlim',[0 8])
legend('Lorenz','NN')

%% Identify Transitions from One Lobe to Another (rho = 28)
rho = r2; % rho = 28
Lorenz = Lorenz2; % Lorenz equations corresponding to rho = 28

X_input = []; X_output = []; Y_input = []; Y_output = [];
x0 = 30*(rand(3,1)-0.5);

[t,y] = ode45(Lorenz,t,x0);
X_input = [X_input; y(1:end-1,1)]; X_output = [X_output; y(2:end,1)];
Y_input = [Y_input; y(1:end-1,2)]; Y_output = [Y_output; y(2:end,2)];

figure(7) % viewing two lobes with a line to show the transition
plot(X_input,Y_input), hold on
plot(-20:20,2.*[-20:20],'Linewidth',[2]), xlabel('X'), ylabel('Y')
title('Transition Line Between Lobes')

% determining the two lobes (setting the values to be 1 or -1)
lobe = [];
for i = 1:length(X_input)
    if Y_input(i) > 2*X_input(i)
        lobe = [lobe; 1];
    else
        lobe = [lobe; -1];
    end
end

% finding the labels for the transitions between lobes (when the labels
% jump from 0 to a large value, we know that it transitioned between lobes)
label = [1]; temp = 0;
for i = length(lobe):-1:2
    if sign(lobe(i)*lobe(i-1)) == -1 % if the data transitioned to a new lobe
        temp = 0; % reset temporary value
        label = [temp; label]; % set label to 0
    else % if the data does not transition to a new lobe
        temp = temp+1; % keep increasing temporary value
        label = [temp; label];
    end
end

% Training NN
Train = [X_output(1:length(label)) Y_output(1:length(label))]; % combining the output sets for training

net0 = feedforwardnet([10 10 10]);
net0.layers{1}.transferFcn = 'logsig';
net0.layers{2}.transferFcn = 'radbas';
net0.layers{3}.transferFcn = 'purelin';
net0 = train(net0,Train.',label.');

% running ODE solution through NN
x0 = 30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
pre_label = y(:,1:2).'; % taking only the x & y directions from ode45
prediction = net0(pre_label); % getting prediction labels

% plotting the predictions
figure(8)
plot(prediction.','r','Linewidth',[2]), hold on
plot(label,'b','Linewidth',[2]), legend('Prediction','True Label')
title('Predictions of Lobe Transition')