%% Part 1: Handel
clear all; close all; clc

% loading handel signal
load handel
v = y'/2;

% plotting handel signal
%plot((1:length(v))/Fs,v);
%xlabel('Time [sec]');
%ylabel('Amplitude');
%title('Signal of Interest, v(n)');

% playing audio
%p8 = audioplayer(v,Fs);
%playblocking(p8);

v = v(1:end-1); %first and last points of v
l = length(v); %setting the length of v

% setting the time and frequency domains
t0 = (0:l)/Fs;
t = t0(1:end-1);
L = (l-1)/Fs;
k = (2*pi/L)*[0:l/2-1, -l/2:-1];
ks = fftshift(k);

% setting multiple window widths
%w = 0.01;
%w = 1;
%w = 100;
w = 1000;

% Setting different time steps
%n = 10;
n = 100;
%n = 250;
t2 = linspace(0,t(end-1),n); %time discretization

% Creating 3 different Gabor filters: Gaussian, Mexican Hat, and Shannon
filter = {@(x) exp(-w*(x).^2), @(x) (1-(x/w).^2).*exp(-((x/w).^2)/2), @(x) (x>-w/2 & x<w/2)};

% pre-setting a spectrogram for 3 different Gabor filter
s1 = zeros(length(t2),l);
s2 = zeros(length(t2),l);
s3 = zeros(length(t2),l);

for j=1:length(t2)
    % Taking the 3 different Gabor filters
    g1 = filter{1}(t-t2(j)); %Gaussian
    g2 = filter{2}(t-t2(j)); %Mexican Hat
    g3 = filter{3}(t-t2(j)); %Shannon
    
    % applying Gabor transforms to signal
    vg1 = g1.*v; %Gaussian
    vg2 = g2.*v; %Mexican Hat
    vg3 = g3.*v; %Shannon
    
    % Fourier transform of filtered signal
    vgt1 = fft(vg1); %Gaussian
    vgt2 = fft(vg2); %Mexican Hat
    vgt3 = fft(vg3); %Shannon
    
    % Storing the data as rows in the spectrogram matrices
    s1(j,:) = abs(fftshift(vgt1)); %Gaussian
    s2(j,:) = abs(fftshift(vgt2)); %Mexican Hat
    s3(j,:) = abs(fftshift(vgt3)); %Shannon
end

% plotting the spectrograms
%figure()
%pcolor(t2,ks,s1.'), shading interp %Gaussian
%pcolor(t2,ks,s2.'), shading interp %Mexican Hat
%pcolor(t2,ks,s3.'), shading interp %Shannon
%colormap('hot'), xlabel('Time [sec]'), ylabel('Frequency [Hz]')

%% Part 2.1: Mary Had A Little Lamb (Piano)
clear all; close all; clc

% plotting & playing mhall
tr_piano = 16; % record time in seconds
y = audioread('music1.wav'); Fs = length(y)/tr_piano;
%plot((1:length(y))/Fs,y);
%xlabel('Time [sec]'); ylabel('Amplitude');
%title('Mary had a little lamb (piano)'); drawnow
%p8 = audioplayer(y,Fs); playblocking(p8);

% setting the time and frequency domains
y = y.';
l = length(y);
t0 = linspace(0,tr_piano,l+1);
t = t0(1:end-1);
k = (2*pi/tr_piano)*[0:l/2-1, -l/2:-1];
ks = fftshift(k);

% generate spectrogram
w = 100; %width of Gabor filter
n = 100; %number of time-steps
t2 = linspace(0,t(end-1),n); %time discretization
s = zeros(length(t2),l); %pre-setting spectrogram

for j=1:length(t2)
    filter = exp(-w*(t-t2(j)).^2); %Gabor filter (Gaussian)
    yf = filter.*y; %Applying filter to piano signal
    yft = fft(yf); %Fourier transform of filtered signal
    s(j,:) = abs(fftshift(yft)); %Storing data as rows in the spectrogram matrix
end

% plotting spectrogram
%figure()
%pcolor(t2,ks,log(s.'+1)), shading interp
%axis([0 15 1500 2500]) %zooming in on the score
%colormap('hot'), xlabel('Time [sec]'), ylabel('Frequency [Hz]')
%title('Mary had a little lamb (Piano) Score');

%% Part 2.2: Mary Had A Little Lamb (Recorder)
clear all; close all; clc

% plotting & playing mhall
%figure(2)
tr_rec = 14; % record time in seconds
y = audioread('music2.wav'); Fs = length(y)/tr_rec;
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (recorder)');
%p8 = audioplayer(y,Fs); playblocking(p8);

% setting the time and frequency domains
y = y.';
l = length(y);
t0 = linspace(0,tr_rec,l+1);
t = t0(1:end-1);
k = (2*pi/tr_rec)*[0:l/2-1, -l/2:-1];
ks = fftshift(k);

% generate spectrogram
w = 100; %width of Gabor filter
n = 100; %number of time-steps
t2 = linspace(0,t(end-1),n); %time discretization
s = zeros(length(t2),l); %pre-setting spectrogram

for j=1:length(t2)
    filter = exp(-w*(t-t2(j)).^2); %Gabor filter (Gaussian)
    yf = filter.*y; %Applying filter to recorder signal
    yft = fft(yf); %Fourier transform of filtered signal
    s(j,:) = abs(fftshift(yft)); %Storing data as rows in the spectrogram matrix
end

% plotting spectrogram
%figure()
%pcolor(t2,ks,log(s.'+1)), shading interp
%axis([0 14 4000 8000]) %zooming in on the score
%colormap('hot'), xlabel('Time [sec]'), ylabel('Frequency [Hz]')
%title('Mary had a little lamb (Recorder) Score');