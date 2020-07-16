clear all; close all; clc

dt = 2; % time-step (2 years)
Year = 1845:dt:1903; % years (from 1845 to 1903)

% hare & lynx populations
Hare = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 ...
    100 92 70 10 11 137 137 18 22 52 83 18 10 9 65];
Lynx = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 ...
    34 45 40 15 15 60 80 26 18 37 50 35 12 12 25];

% interpolating the data
dt_new = 0.2; % new time-step
t = 1845:dt_new:1903; % query points for interpolation

x1 = spline(Year,Hare,t); % hare population
x2 = spline(Year,Lynx,t); % lynx population

X = [x1; x2]; % combining the data set into one matrix
t_space = 0:dt_new:t(end)-t(1); % adjusting data to go from 0 to 1903-1845

% plotting the interpolated data set
figure(1)
plot(Year,Hare,'ro',Year,Lynx,'bo','Linewidth',[2]), hold on
plot(t,abs(x1),'r',t,abs(x2),'b','Linewidth',[2])
xlabel('Year'), ylabel('Populations'), axis([1845 1903 0 180])
legend('Hare','Lynx','Hare Interp','Lynx Interp','Location','NorthEast')
title('Interpolated Data Set')

%% Part 1 (DMD Model)

X1 = X(:,1:end-1); % the 1 through n-1 terms
X2 = X(:,2:end); % the 2 through n terms

[U2,Sigma2,V2] = svd(X1,'econ'); % computing the svd of the data

% plotting the svs to find the rank (r = 2)
figure(), plot(diag(Sigma2)/sum(diag(Sigma2)),'ro'), axis([0 2.5 0 1])

% computing eigenvalues
r = 2; % the rank of the svd
U = U2(:,1:r); Sigma = Sigma2(1:r,1:r); V = V2(:,1:r); % truncating (r=2)
Atilde = (U')*X2*V/Sigma;
[W,D] = eig(Atilde);
Phi = X2*V/Sigma*W;

mu = diag(D); % taking the eigenvalues
omega = log(mu)/dt_new;

u0 = X(:,1); % initial conditions
y0 = Phi\u0; % pseudo-inverse of initial conditions

% finding the dmd modes
u_modes = zeros(r,length(t_space));
for iter = 1:length(t_space)
    u_modes(:,iter) =(y0.*exp(omega*t_space(iter)));
end
u_dmd = Phi*u_modes; % computing the dmd

hare_dmd1 = real(u_dmd(1,:)); % hare population dmd
lynx_dmd1 = real(u_dmd(2,:)); % lynx population dmd

% plotting the dmd of the populations versus the original data
figure(2)
plot(t_space,abs(x1),'r',t_space,abs(x2),'b','Linewidth',[2]), hold on
plot(t_space,abs(hare_dmd1),'g',t_space,abs(lynx_dmd1),'m','Linewidth',[2])
xlabel('Time [years]'), ylabel('Populations')
legend('Hare','Lynx','Hare DMD','Lynx DMD','Location','NorthEast')
title('DMD Model of Populations')

%% Part 2 (Time-Delay DMD Model)

embeddings = 25; % number of time-delay embeddings

H = []; % hankel matrix
for i = 1:embeddings
    h = x1(i:end-(embeddings-i)); % row from hare population
    l = x2(i:end-(embeddings-i)); % row from lynx population
    H = [H; h; l]; % storing hare & lynx in 2 rows of the hankel matrix
end

X1 = H(:,1:end-1); % the 1 through n-1 terms
X2 = H(:,2:end); % the 2 through n terms

[U2,Sigma2,V2] = svd(X1,'econ'); % computing svd of hankel matrix

% plotting the svs to find the rank (r = 6)
figure(), plot(diag(Sigma2)/sum(diag(Sigma2)),'ro','Linewidth',[2])
xlabel('Singular Values'), ylabel('Variance'), title('Singular Value Spectrum')

% computing eigenvalues
r = 6; % the rank of the svd
U = U2(:,1:r); Sigma = Sigma2(1:r,1:r); V = V2(:,1:r); % truncating (r=6)
Atilde = (U')*X2*V/Sigma;
[W,D] = eig(Atilde);
Phi = X2*V/Sigma*W;

mu = diag(D); % taking the eigenvalues
omega = log(mu)/dt_new;

u0 = H(:,1); % initial conditions
y0 = Phi\u0; % pseudo-inverse of initial conditions

% finding the dmd modes
u_modes = zeros(r,length(t_space));
for iter = 1:length(t_space)
    u_modes(:,iter) =(y0.*exp(omega*t_space(iter)));
end
u_dmd = Phi*u_modes; % computing the dmd

hare_dmd2 = real(u_dmd(1,:)); % hare population dmd
lynx_dmd2 = real(u_dmd(2,:)); % lynx population dmd

% plotting the dmd of the populations versus the original data
figure(3)
plot(t_space,abs(x1),'r',t_space,abs(x2),'b','Linewidth',[2]), hold on
plot(t_space,abs(hare_dmd2),'g',t_space,abs(lynx_dmd2),'m','Linewidth',[2])
xlabel('Time [years]'), ylabel('Populations')
legend('Hare','Lynx','Hare DMD','Lynx DMD','Location','NorthEast')
title('Time-Delay DMD Model of Populations')

%% Part 3 (Lotka-Volterra)

b = 1; p = 0; r = -0.001; d = 1; % initial guess for LV parameters
[t0,ut] = ode45('rhs_dyn',t_space,X,b,p,r,d); % LV model rhs

x1_lv = ut(:,292:end); % hare population
x2_lv = ut(:,1:291); % lynx population

n = length(t0);
% computing the derivative using central difference
for j = 2:n-1
    x1dot(j-1) = (x1_lv(j+1) - x1_lv(j-1))/(2*dt_new); % hare population
    x2dot(j-1) = (x2_lv(j+1) - x2_lv(j-1))/(2*dt_new); % lynx population
end

% setting data to be the same size as its derivative
x1s = x1_lv(2:n-1); % hare population
x2s = x2_lv(2:n-1); % lynx population

% building library matrix A with x1, x2, & x1*x2
A = [x1s; x2s; x1s.*x2s];

% solving Ax = b using pinv
xi1 = (pinv(A).')*(x1dot.'); % hare population
xi2 = (pinv(A).')*(x2dot.'); % lynx population

% plotting the coefficients from pinv
figure(4)
subplot(2,1,1), bar(xi1) % hare population
xlabel('Functions'), ylabel('Coefficient')
title('Weights for Hare Population (Pinv)')
subplot(2,1,2), bar(xi2) % lynx population
xlabel('Functions'), ylabel('Coefficient')
title('Weights for Lynx Population (Pinv)')

% computing best fit for hare population
hf = xi1(1)*x1 + xi1(2)*x2 + xi1(3)*x1.*x2;

% computing best fit for lynx population
lf = xi2(1)*x1 + xi2(2)*x2 + xi2(3)*x1.*x2;

% plotting the LV model fit versus the original data
figure(5)
plot(t_space,abs(x1),'r',t_space,abs(x2),'b','Linewidth',[2]), hold on
plot(t_space,abs(hf),'g',t_space,abs(lf),'m','Linewidth',[2])
xlabel('Time [years]'), ylabel('Populations')
legend('Hare','Lynx','Hare LV','Lynx LV','Location','NorthEast')
title('Lotka-Volterra Model of Populations (Pinv)')

%% Part 4 (Best Fit Using Lasso & Ridge Regression)

% LASSO

bl = 1; pl = 0; rl = -0.005; dl = 0; % initial guess for lasso parameters
[t0_l,ut_l] = ode45('rhs_dyn',t_space,X,bl,pl,rl,dl); % LV model (lasso)

x1_l = ut_l(:,292:end); % hare population
x2_l = ut_l(:,1:291); % lynx population

n_l = length(t0_l);
% computing the derivative using central difference
for j = 2:n_l-1
    x1dot_l(j-1) = (x1_l(j+1) - x1_l(j-1))/(2*dt_new); % hare population
    x2dot_l(j-1) = (x2_l(j+1) - x2_l(j-1))/(2*dt_new); % lynx population
end

% setting data to be the same size as its derivative
x1s_l = x1_l(2:n_l-1); % hare population
x2s_l = x2_l(2:n_l-1); % lynx population

% building library matrix A
A_l = [x1s_l; x2s_l; x1s_l.^2; x1s_l.*x2s_l; x2s_l.^2; x1s_l.^3; ...
    (x1s_l.^2).*x2s_l; (x2s_l.^2).*x1s_l; x2s_l.^3; sin(x1s_l); ...
    cos(x1s_l); sin(x2s_l); cos(x2s_l); sin(x1s_l).*cos(x2s_l); ...
    cos(x1s_l).*sin(x2s_l)];

% solving Ax = b using lasso method
[xi1_l,~] = lasso(A_l',x1dot_l.','Lambda',0.025); % hare population
[xi2_l,~] = lasso(A_l',x2dot_l.','Lambda',0.025); % lynx population

% plotting the coefficients from lasso
figure(6)
subplot(2,1,1), bar(xi1_l) % hare population
xlabel('Functions'), ylabel('Coefficient')
title('Weights for Hare Population (Lasso)')
subplot(2,1,2), bar(xi2_l) % lynx population
xlabel('Functions'), ylabel('Coefficient')
title('Weights for Lynx Population (Lasso)')

% computing best fit for hare population
hf_l = xi1_l(1)*x1 + xi1_l(2)*x2 + xi1_l(3)*x1.^2 + xi1_l(4)*(x1.*x2) ...
    + xi1_l(5)*x2.^2 + xi1_l(6)*x1.^3 + xi1_l(7)*(x1.^2).*x2 ...
    + xi1_l(8)*(x2.^2).*x1 + xi1_l(9)*x2.^3 + xi1_l(10)*sin(x1) ...
    + xi1_l(11)*cos(x1) + xi1_l(12)*sin(x2) + xi1_l(13)*cos(x2) ...
    + xi1_l(14)*(sin(x1).*cos(x2)) + xi1_l(15)*(cos(x1).*sin(x2));

% computing best fit for lynx population
lf_l = xi2_l(1)*x1 + xi2_l(2)*x2 + xi2_l(3)*x1.^2 + xi2_l(4)*(x1.*x2) ...
    + xi2_l(5)*x2.^2 + xi2_l(6)*x1.^3 + xi2_l(7)*(x1.^2).*x2 ...
    + xi2_l(8)*(x2.^2).*x1 + xi2_l(9)*x2.^3 + xi2_l(10)*sin(x1) ...
    + xi2_l(11)*cos(x1) + xi2_l(12)*sin(x2) + xi2_l(13)*cos(x2) ...
    + xi2_l(14)*(sin(x1).*cos(x2)) + xi2_l(15)*(cos(x1).*sin(x2));

% plotting the LV model fit versus the original data
figure(7)
plot(t_space,abs(x1),'r',t_space,abs(x2),'b','Linewidth',[2]), hold on
plot(t_space,abs(hf_l),'g',t_space,abs(lf_l),'m','Linewidth',[2])
xlabel('Time [years]'), ylabel('Populations')
legend('Hare','Lynx','Hare LV','Lynx LV','Location','NorthEast')
title('Lotka-Volterra Model of Populations (Lasso)')

% RIDGE

br = 1; pr = 0; rr = 0; dr = -0.001; % initial guess for ridge parameters
[t0_r,ut_r] = ode45('rhs_dyn',t_space,X,br,pr,rr,dr); % LV model (ridge)

x1_r = ut_r(:,292:end); % hare population
x2_r = ut_r(:,1:291); % lynx population

n_r = length(t0_l);
% computing the derivative using central difference
for j = 2:n_r-1
    x1dot_r(j-1) = (x1_r(j+1) - x1_r(j-1))/(2*dt_new); % hare population
    x2dot_r(j-1) = (x2_r(j+1) - x2_r(j-1))/(2*dt_new); % lynx population
end

% setting data to be the same size as its derivative
x1s_r = x1_r(2:n_r-1); % hare population
x2s_r = x2_r(2:n_r-1); % lynx population

% building library matrix A
A_r = [x1s_r; x2s_r; x1s_r.^2; x1s_r.*x2s_r; x2s_r.^2; x1s_r.^3; ...
    (x1s_r.^2).*x2s_r; (x2s_r.^2).*x1s_r; x2s_r.^3; sin(x1s_r); ...
    cos(x1s_r); sin(x2s_r); cos(x2s_r); sin(x1s_r).*cos(x2s_r); ...
    cos(x1s_r).*sin(x2s_r)];

% solving Ax = b using ridge method
xi1_r = ridge(x1dot_r.',A_r',0.025); % hare population
xi2_r = ridge(x2dot_r.',A_r',0.025); % lynx population

% plotting the coefficients from ridge
figure(8)
subplot(2,1,1), bar(xi1_r) % hare population
xlabel('Functions'), ylabel('Coefficient')
title('Weights for Hare Population (Ridge)')
subplot(2,1,2), bar(xi2_r) % lynx population
xlabel('Functions'), ylabel('Coefficient')
title('Weights for Lynx Population (Ridge)')

% computing best fit for hare population
hf_r = xi1_r(1)*x1 + xi1_r(2)*x2 + xi1_r(3)*x1.^2 + xi1_r(4)*(x1.*x2) ...
    + xi1_r(5)*x2.^2 + xi1_r(6)*x1.^3 + xi1_r(7)*(x1.^2).*x2 ...
    + xi1_r(8)*(x2.^2).*x1 + xi1_r(9)*x2.^3 + xi1_r(10)*sin(x1) ...
    + xi1_r(11)*cos(x1) + xi1_r(12)*sin(x2) + xi1_r(13)*cos(x2) ...
    + xi1_r(14)*(sin(x1).*cos(x2)) + xi1_r(15)*(cos(x1).*sin(x2));

% computing best fit for lynx population
lf_r = xi2_r(1)*x1 + xi2_r(2)*x2 + xi2_r(3)*x1.^2 + xi2_r(4)*(x1.*x2) ...
    + xi2_r(5)*x2.^2 + xi2_r(6)*x1.^3 + xi2_r(7)*(x1.^2).*x2 ...
    + xi2_r(8)*(x2.^2).*x1 + xi2_r(9)*x2.^3 + xi2_r(10)*sin(x1) ...
    + xi2_r(11)*cos(x1) + xi2_r(12)*sin(x2) + xi2_r(13)*cos(x2) ...
    + xi2_r(14)*(sin(x1).*cos(x2)) + xi2_r(15)*(cos(x1).*sin(x2));

% plotting the LV model fit versus the original data
figure(9)
plot(t_space,abs(x1),'r',t_space,abs(x2),'b','Linewidth',[2]), hold on
plot(t_space,abs(hf_r),'g',t_space,abs(lf_r),'m','Linewidth',[2])
xlabel('Time [years]'), ylabel('Populations')
legend('Hare','Lynx','Hare LV','Lynx LV','Location','NorthEast')
title('Lotka-Volterra Model of Populations (Ridge)')

%% Part 5 (Compute KL Divergence)

Y = [abs(x1); abs(x2)]; % truth model
Y1 = [abs(hare_dmd1); abs(lynx_dmd1)]; % dmd model (part 1)
Y2 = [abs(hare_dmd2); abs(lynx_dmd2)]; % time-delay dmd model (part 2)
Y3 = [abs(hf); abs(lf)]; % pinv model (part 3)
Y4 = [abs(hf_l); abs(lf_l)]; % lasso model (part 4.1)
Y5 = [abs(hf_r); abs(lf_r)]; % ridge model (part 4.2)

% generate PDFs
fa = hist(Y(1,:),t_space)+dt_new; fb = hist(Y(2,:),t_space)+dt_new; f = [fa; fb]; % truth model
g1a = hist(Y1(1,:),t_space)+dt_new; g1b = hist(Y1(2,:),t_space)+dt_new; g1 = [g1a; g1b]; % dmd model
g2a = hist(Y2(1,:),t_space)+dt_new; g2b = hist(Y2(2,:),t_space)+dt_new; g2 = [g2a; g2b]; % time-delay dmd model
g3a = hist(Y3(1,:),t_space)+dt_new; g3b = hist(Y3(2,:),t_space)+dt_new; g3 = [g3a; g3b]; % pinv model
g4a = hist(Y4(1,:),t_space)+dt_new; g4b = hist(Y4(2,:),t_space)+dt_new; g4 = [g4a; g4b]; % lasso model
g5a = hist(Y5(1,:),t_space)+dt_new; g5b = hist(Y5(2,:),t_space)+dt_new; g5 = [g5a; g5b]; % ridge model

% normalize data
f = [f(1,:)/trapz(t_space,f(1,:)); f(2,:)/trapz(t_space,f(2,:))]; % truth model
g1 = [g1(1,:)/trapz(t_space,g1(1,:)); g1(2,:)/trapz(t_space,g1(2,:))]; % dmd model
g2 = [g2(1,:)/trapz(t_space,g2(1,:)); g2(2,:)/trapz(t_space,g2(2,:))]; % time-delay dmd model
g3 = [g3(1,:)/trapz(t_space,g3(1,:)); g3(2,:)/trapz(t_space,g3(2,:))]; % pinv model
g4 = [g4(1,:)/trapz(t_space,g4(1,:)); g4(2,:)/trapz(t_space,g4(2,:))]; % lasso model
g5 = [g5(1,:)/trapz(t_space,g5(1,:)); g5(2,:)/trapz(t_space,g5(2,:))]; % ridge model

% plotting normalized PDF's
figure(10)
plot(t_space,f(1,:),'r',t_space,f(2,:),'b','Linewidth',[1.5]), hold on % truth model
plot(t_space,g1(1,:),'g',t_space,g1(2,:),'m','Linewidth',[1.5]), hold on % dmd model
plot(t_space,g2(1,:),'k',t_space,g2(2,:),'y','Linewidth',[1.5]), hold on % time-delay dmd model
plot(t_space,g3(1,:),'r--',t_space,g3(2,:),'b--','Linewidth',[1.5]), hold on % pinv model
plot(t_space,g4(1,:),'g--',t_space,g4(2,:),'m--','Linewidth',[1.5]), hold on % lasso model
plot(t_space,g5(1,:),'k--',t_space,g5(2,:),'y--','Linewidth',[1.5]) % ridge model
xlabel('Time [years]'), ylabel('Probability Distribution')
legend('Hare','Lynx','DMD (Hare)','DMD (Lynx)','Time-Delay DMD (Hare)',...
    'Time-Delay DMD (Lynx)','Pinv (Hare)','Pinv (Lynx)','Lasso (Hare)',...
    'Lasso (Lynx)','Ridge (Hare)','Ridge (Lynx)','Location','EastOutside')
title('Normalized PDFs of the Models'), axis([0 58 0 0.15])

% compute integrand
Int1a = f(1,:).*log(f(1,:)./g1(1,:)); Int1b = f(2,:).*log(f(2,:)./g1(2,:)); Int1 = [Int1a; Int1b]; % dmd model
Int2a = f(1,:).*log(f(1,:)./g2(1,:)); Int2b = f(2,:).*log(f(2,:)./g2(2,:)); Int2 = [Int2a; Int2b]; % time-delay dmd model
Int3a = f(1,:).*log(f(1,:)./g3(1,:)); Int3b = f(2,:).*log(f(2,:)./g3(2,:)); Int3 = [Int3a; Int3b]; % pinv model
Int4a = f(1,:).*log(f(1,:)./g4(1,:)); Int4b = f(2,:).*log(f(2,:)./g4(2,:)); Int4 = [Int4a; Int4b]; % lasso model
Int5a = f(1,:).*log(f(1,:)./g5(1,:)); Int5b = f(2,:).*log(f(2,:)./g5(2,:)); Int5 = [Int5a; Int5b]; % ridge model

% KL divergence
I1a = trapz(t_space,Int1(1,:)); I1b = trapz(t_space,Int1(2,:)); I1 = [I1a; I1b]; % dmd model
I2a = trapz(t_space,Int2(1,:)); I2b = trapz(t_space,Int2(2,:)); I2 = [I2a; I2b]; % time-delay dmd model
I3a = trapz(t_space,Int3(1,:)); I3b = trapz(t_space,Int3(2,:)); I3 = [I3a; I3b]; % pinv model
I4a = trapz(t_space,Int4(1,:)); I4b = trapz(t_space,Int4(2,:)); I4 = [I4a; I4b]; % lasso model
I5a = trapz(t_space,Int5(1,:)); I5b = trapz(t_space,Int5(2,:)); I5 = [I5a; I5b]; % ridge model

% plotting KL divergence
figure(11)
bar_x = categorical({'Hare','Lynx'}); % x-axis of bar graph
bar_y = [I1(1,:) I2(1,:) I3(1,:) I4(1,:) I5(1,:);...
    I1(2,:) I2(2,:) I3(2,:) I4(2,:) I5(2,:)]; % y-axis of bar graph
bar(bar_x, bar_y), ylabel('Magnitude'), title('KL Divergence of the Best Fit Models')
legend('DMD','Time-Delay DMD','Pinv','Lasso','Ridge','Location','NorthEast')

%% Part 6 (Compare AIC & BIC Scores)

% loading PDF's
Z1 = g2; % normalized PDF of time-delay dmd model
Z2 = g3; % normalized PDF of pinv model
Z3 = g5; % normalized PDF of ridge model

% computing log-likelihood
L1a = sum(log(Z1(1,:))); L1b = sum(log(Z1(2,:))); L1 = [L1a; L1b]; % time-delay dmd model
L2a = sum(log(Z2(1,:))); L2b = sum(log(Z2(2,:))); L2 = [L2a; L2b]; % pinv model
L3a = sum(log(Z3(1,:))); L3b = sum(log(Z3(2,:))); L3 = [L3a; L3b]; % ridge model

% number of parameters
k1 = 2; % time-delay dmd model (rank & dt_new)
k2 = 7; % pinv model (b, r, d, dt_new, & 3 A coefficients)
k3 = 19; % ridge model (b, d, dt_new, lambda, & 15 A coefficients)

% computing AIC scores
aic1a = 2*k1 - 2*L1(1); aic1b = 2*k1 - 2*L1(2); aic1 = [aic1a; aic1b]; % time-delay dmd model
aic2a = 2*k2 - 2*L2(1); aic2b = 2*k2 - 2*L2(2); aic2 = [aic2a; aic2b]; % pinv model
aic3a = 2*k3 - 2*L3(1); aic3b = 2*k3 - 2*L3(2); aic3 = [aic3a; aic3b]; % ridge model

% plotting AIC scores
figure(12)
aic_x = categorical({'Hare','Lynx'}); % x-axis of bar graph
aic_y = [aic1(1) aic2(1) aic3(1); aic1(2) aic2(2) aic3(2)]; % y-axis of bar graph
bar(aic_x, aic_y), ylabel('Scores'), title('AIC Scores')
legend('Time-Delay DMD','Pinv','Ridge','Location','EastOutside')

n = length(t_space); % sample size

% computing BIC scores
bic1a = k1*log(n) - 2*L1(1); bic1b = k1*log(n) - 2*L1(2); bic1 = [bic1a; bic1b]; % time-delay dmd model
bic2a = k2*log(n) - 2*L2(1); bic2b = k2*log(n) - 2*L2(2); bic2 = [bic2a; bic2b]; % pinv model
bic3a = k3*log(n) - 2*L3(1); bic3b = k3*log(n) - 2*L3(2); bic3 = [bic3a; bic3b]; % ridge model

% plotting BIC scores
figure(13)
bic_x = categorical({'Hare','Lynx'}); % x-axis of bar graph
bic_y = [bic1(1) bic2(1) bic3(1); bic1(2) bic2(2) bic3(2)]; % y-axis of bar graph
bar(bic_x, bic_y), ylabel('Scores'), title('BIC Scores')
legend('Time-Delay DMD','Pinv','Ridge','Location','EastOutside')