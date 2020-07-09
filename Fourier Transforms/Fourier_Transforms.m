clear all; close all; clc
load Testdata

L = 15; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L, L, n+1); x = x2(1:n); y = x; z = x; % spatial components
k = (2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks = fftshift(k); % frequency components of fft
Uave = zeros(1,n); % 3-dimensional matrix for the average spectrum

[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(ks,ks,ks);

% averaging the spectrum
for j=1:20
    Un(:,:,:) = reshape(Undata(j,:),n,n,n); % getting the data from the jth row
    Uave = Uave+fftshift(fftn(Un(:,:,:))); % adding the fft data to the average spectrum

    % plotting the spectrum
    %close all, isosurface(Kx,Ky,Kz,abs(Uave)/max(abs(Uave(:))),0.6)
    %axis([-6 6 -6 6 -6 6]), grid on
    %title('Frequency Spectrum'), xlabel('kx'), ylabel('ky'), zlabel('kz'), drawnow
    %pause(1)
end

% finding the center of the frequency
[m,I] = max(abs(Uave(:)));
[fx,fy,fz] = ind2sub(size(Uave),I);
Center = [Kx(fx,fy,fz),Ky(fx,fy,fz),Kz(fx,fy,fz)]; % the center frequency
kx = Center(1); % x-coordinate of the center frequency
ky = Center(2); % y-coordinate of the center frequency
kz = Center(3); % z-coordinate of the center frequency

% filtering the center frequency
filter = exp(-0.2*((Kx-kx).^2+(Ky-ky).^2+(Kz-kz).^2));
%isosurface(Kx,Ky,Kz,filter,0.6)
%axis([-6 6 -6 6 -6 6]), grid on
%title('3D Gaussian Filter'), xlabel('kx'), ylabel('ky'), zlabel('kz'), drawnow

Uave1 = zeros(1,n); % 3-dimensional matrix for the average spectrum
marble = zeros(3,20); % 2-dimensional matrix representing the path of the marble

% applying the filter to the data
for jj=1:20
    Un(:,:,:) = reshape(Undata(jj,:),n,n,n); % getting the data from the jth row
    Ut = fftshift(fftn(Un(:,:,:))); % taking the fft of the data
    Unft = filter.*Ut; % applying the filter
    Unf = ifftn(Unft); % reversing the transform
    
    % plotting the path of the filter
    Uave1 = Uave1+Unf; % average of the filtered data
    %close all, isosurface(Kx,Ky,Kz,abs(Uave1)/max(abs(Uave1(:))),0.6)
    %axis([-6 6 -6 6 -6 6]), grid on
    %title('Path through the Frequency Domain'), xlabel('kx'), ylabel('ky'), zlabel('kz'), drawnow
    %pause(1)
    
    % finding the path of the marble in the spatial domain
    [m1,I1] = max(abs(Unf(:)));
    [fx1,fy1,fz1] = ind2sub(size(Unf),I1);
    marble(1,jj) = X(fx1,fy1,fz1); % x-coordinate of the marble
    marble(2,jj) = Y(fx1,fy1,fz1); % y-coordinate of the marble
    marble(3,jj) = Z(fx1,fy1,fz1); % z-coordinate of the marble
end

% plotting the path of the marble
figure()
plot3(marble(1,:),marble(2,:),marble(3,:),'m-o'), grid on
title('The Path of the Marble'), xlabel('x'), ylabel('y'), zlabel('z')