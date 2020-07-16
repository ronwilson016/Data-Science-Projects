clear all; close all; clc

% loading the data
S = load('BZ.mat');
Y = S.BZ_tensor;

[m,n,k] = size(Y); % x vs y vs time data

% viewing the data
%for j = 1:k
%    A = Y(:,:,j);
%    pcolor(A), shading interp, pause(0.2)
%end

%% Part 1 (DMD Model)

dx = 1;
x = 0:dx:n; % spacial domain

frames = [1 500 1000 k]; % taking four frames of the data
i = 4; % select specific frame

X = Y(:,:,frames(i)); % data matrix
X1 = X(:,1:end-1); % the 1 through n-1 terms
X2 = X(:,2:end); % the 2 through n terms

[U2,Sigma2,V2] = svd(X1,'econ'); % computing the svd of the data

% plotting the svs to find the rank (r = 10)
figure(1), plot(diag(Sigma2)/sum(diag(Sigma2)),'ro','Linewidth',[2])

% computing eigenvalues
r = 10; % the rank of the svd
U = U2(:,1:r); Sigma = Sigma2(1:r,1:r); V = V2(:,1:r); % truncating (r=10)
Atilde = (U')*X2*V/Sigma;
[W,D] = eig(Atilde);
Phi = X2*V/Sigma*W;

mu = diag(D); % taking the eigenvalues
omega = log(mu)/dx;

u0 = X(:,1); % initial conditions
y0 = Phi\u0; % pseudo-inverse of initial conditions

% finding the dmd modes
u_modes = zeros(r,length(x));
for iter = 1:length(x)
    u_modes(:,iter) =(y0.*exp(omega*x(iter)));
end
u_dmd = Phi*u_modes; % computing the dmd

% plotting dmd model & original data
figure(2) % viewing dmd model
pcolor(real(u_dmd)), shading interp
xlabel('X'), ylabel('Y'), title('DMD Model')
figure(3) % viewing original frame
pcolor(X), shading interp
xlabel('X'), ylabel('Y'), title('Original Data')

%% Part 2 (Time-Delay DMD Model)

dx = 1;
x = 0:dx:n; % spacial domain

frames = [1 500 1000 k]; % taking four frames of the data
i = 4; % select specific frame

X = Y(:,:,frames(i)); % data matrix
embeddings = 25; % number of time-delay embeddings

H = []; % hankel matrix
for i = 1:embeddings
    h = X(:,i:end-(embeddings-i));
    H = [H; h]; % storing data as rows of the hankel matrix
end

X1 = H(:,1:end-1); % the 1 through n-1 terms
X2 = H(:,2:end); % the 2 through n terms

[U2,Sigma2,V2] = svd(X1,'econ'); % computing the svd of the data

% plotting the svs to find the rank (r = 10)
figure(4), plot(diag(Sigma2)/sum(diag(Sigma2)),'ro','Linewidth',[2])

% computing eigenvalues
r = 10; % the rank of the svd
U = U2(:,1:r); Sigma = Sigma2(1:r,1:r); V = V2(:,1:r); % truncating (r=10)
Atilde = (U')*X2*V/Sigma;
[W,D] = eig(Atilde);
Phi = X2*V/Sigma*W;

mu = diag(D); % taking the eigenvalues
omega = log(mu)/dx;

u0 = H(:,1); % initial conditions
y0 = Phi\u0; % pseudo-inverse of initial conditions

% finding the dmd modes
u_modes = zeros(r,length(x));
for iter = 1:length(x)
    u_modes(:,iter) =(y0.*exp(omega*x(iter)));
end
u_dmd = Phi*u_modes; % computing the dmd

% plotting dmd model & original data
figure(5) % viewing dmd model of first mode only
pcolor(real(u_dmd(1:m,:))), shading interp
xlabel('X'), ylabel('Y'), title('Time-Delay DMD Model')
figure(6) % viewing original frame
pcolor(X), shading interp
xlabel('X'), ylabel('Y'), title('Original Data')