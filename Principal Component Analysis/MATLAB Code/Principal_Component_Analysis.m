%% Test 1: Ideal Case (camN_1 where N = 1,2,3)
clear all; close all; clc

% loading video files
load('cam1_1.mat'); %camera 1
load('cam2_1.mat'); %camera 2
load('cam3_1.mat'); %camera 3

% finding initial x,y coordinates from each video by clicking on the flashlight
ind1 = zeros(2,size(vidFrames1_1,4)); %camera 1
ind2 = zeros(2,size(vidFrames2_1,4)); %camera 1
ind3 = zeros(2,size(vidFrames3_1,4)); %camera 3
%camera 1
figure()
imshow(vidFrames1_1(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x1,y1] = ginput(1);
ind1(:,1) = [y1; x1]; %setting first position of the flashlight
%camera 2
figure()
imshow(vidFrames2_1(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x2,y2] = ginput(1);
ind2(:,1) = [y2; x2]; %setting first position of the flashlight
%camera 3
figure()
imshow(vidFrames3_1(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x3,y3] = ginput(1);
ind3(:,1) = [y3; x3]; %setting first position of the flashlight

close all; %close the cameras after finding the initial position

rw = 15; %size of row windows
cw = 15; %size of column windows
tr = 0; %temporary row size
tc = 0; %temporary column size
m1 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 1
m2 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 2
m3 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 3

% window for camera 1
for jj=2:size(vidFrames1_1,4)
    r1 = ind1(1,jj-1); %prior row location of flashlight
    c1 = ind1(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r1-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r1-1+(r1 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r1+rw) > size(vidFrames1_1,1) %if the window is larger than the size of the frame
        tr = size(vidFrames1_1,1)-r1+(r1 == size(vidFrames1_1,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c1-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c1-1+(c1 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c1+cw) > size(vidFrames1_1,1) %if the window is larger than the size of the frame
        tc = size(vidFrames1_1,1)-c1+(c1 == size(vidFrames1_1,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m1 = vidFrames1_1((r1-tr):(r1+tr),(c1-tc):(c1+tc),:,jj); %windowing the flashlight
    tm1 = (sum(m1,3) == max(max(sum(m1,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm1); %place the maximum intensity pixel in the (1,1) position
    ind1(1,jj) = r(1)+(r1-tr)-1; %extract the location of the pixel
    ind1(2,jj) = c(1)+(c1-tc)-1; %extract the location of the pixel
end
% window for camera 2
for jj=2:size(vidFrames2_1,4)
    r2 = ind2(1,jj-1); %prior row location of flashlight
    c2 = ind2(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r2-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r2-1+(r2 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r2+rw) > size(vidFrames2_1,1) %if the window is larger than the size of the frame
        tr = size(vidFrames2_1,1)-r2+(r2 == size(vidFrames2_1,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c2-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c2-1+(c2 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c2+cw) > size(vidFrames2_1,1) %if the window is larger than the size of the frame
        tc = size(vidFrames2_1,1)-c2+(c2 == size(vidFrames2_1,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m2 = vidFrames2_1((r2-tr):(r2+tr),(c2-tc):(c2+tc),:,jj); %windowing the flashlight
    tm2 = (sum(m2,3) == max(max(sum(m2,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm2); %place the maximum intensity pixel in the (1,1) position
    ind2(1,jj) = r(1)+(r2-tr)-1; %extract the location of the pixel
    ind2(2,jj) = c(1)+(c2-tc)-1; %extract the location of the pixel
end
% window for camera 3
for jj=2:size(vidFrames3_1,4)
    r3 = ind3(1,jj-1); %prior row location of flashlight
    c3 = ind3(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r3-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r3-1+(r3 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r3+rw) > size(vidFrames3_1,1) %if the window is larger than the size of the frame
        tr = size(vidFrames3_1,1)-r3+(r3 == size(vidFrames3_1,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c3-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c3-1+(c3 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c3+cw) > size(vidFrames3_1,1) %if the window is larger than the size of the frame
        tc = size(vidFrames3_1,1)-c3+(c3 == size(vidFrames3_1,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m3 = vidFrames3_1((r3-tr):(r3+tr),(c3-tc):(c3+tc),:,jj); %windowing the flashlight
    tm3 = (sum(m3,3) == max(max(sum(m3,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm3); %place the maximum intensity pixel in the (1,1) position
    ind3(1,jj) = r(1)+(r3-tr)-1; %extract the location of the pixel
    ind3(2,jj) = c(1)+(c3-tc)-1; %extract the location of the pixel
end

% aligning the 3 cameras to have max and min occur at the same time
for i = 1:2
    ind1(i,:) = (ind1(i,:)-min(ind1(i,:)))/max(ind1(i,:)); %camera 1
    ind2(i,:) = (ind2(i,:)-min(ind2(i,:)))/max(ind2(i,:)); %camera 2
    ind3(i,:) = (ind3(i,:)-min(ind3(i,:)))/max(ind3(i,:)); %camera 3
end

% verifying that the cameras are aligned
%figure()
%plot(ind1(1,2:end)), hold on
%plot(ind2(1,10:end))
%plot(ind3(2,1:end))
%xlabel('Frame'), ylabel('Row Index'), title('Normalized Indices over Time')
%legend('Camera 1', 'Camera 2', 'Camera 3')

% setting all 3 cameras to start at the same time
ind1 = ind1(:,9:end); %camera 1
ind2 = ind2(:,18:end); %camera 2
ind3 = ind3(:,8:end); %camera 3
t = min([size(ind1,2),size(ind2,2),size(ind3,2)]); %minimum number of time-steps
t0 = linspace(0,1,t); %time discretization
X = zeros(6,t); %cropping position matrix to be the same length for all cameras

% filling in the position matrix
X(1,:) = ind1(1,1:t);
X(2,:) = ind1(2,1:t);
X(3,:) = ind2(1,1:t);
X(4,:) = ind2(2,1:t);
X(5,:) = ind3(1,1:t);
X(6,:) = ind3(2,1:t);

X = X-repmat(mean(X,2),1,t); %subtract the mean
[u,s,v] = svd(X/sqrt(t-1)); %finding the SVD
sig = diag(s);
var = sig/sum(sig); %computing variance
Y = u.'*X; %projection of principal components

% PCA plots
figure()
subplot(3,2,1), plot(sig,'ko-') %singular values
xlabel('Modes'), ylabel('Sigma'), title('Singular Values') 
subplot(3,2,2), plot(var*100,'ko-') %percentage of variance
xlabel('Modes'), ylabel('Percentage'), title('Variance'), 
subplot(3,1,2) %projection of PCA
plot(t0,Y(1,:),'k',t0,Y(2,:),'r',t0,Y(3,:),'b',t0,Y(4,:),'c',t0,Y(5,:),'g',t0,Y(6,:),'m','Linewidth',[2])
legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Location','EastOutside')
xlabel('Time'), ylabel('Principal Components'), title('PCA Projection over Time')
subplot(3,1,3) %time behavior
plot(t0,v(1,:),'k',t0,v(2,:),'r',t0,v(3,:),'b',t0,v(4,:),'c',t0,v(5,:),'g',t0,v(6,:),'m')
legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Location','EastOutside')
xlabel('Time'), ylabel('PCA Data'), title('Time Evolution Behavior over PCA Modes')

%% Test 2: Noisy Case (camN_2 where N = 1,2,3)
clear all; close all; clc

% loading cam files
load('cam1_2.mat'); %camera 1
load('cam2_2.mat'); %camera 2
load('cam3_2.mat'); %camera 3

% finding initial x,y coordinates from each video by clicking on the flashlight
ind1 = zeros(2,size(vidFrames1_2,4)); %camera 1
ind2 = zeros(2,size(vidFrames2_2,4)); %camera 1
ind3 = zeros(2,size(vidFrames3_2,4)); %camera 3
%camera 1
figure()
imshow(vidFrames1_2(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x1,y1] = ginput(1);
ind1(:,1) = [y1; x1]; %setting first position of the flashlight
%camera 2
figure()
imshow(vidFrames2_2(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x2,y2] = ginput(1);
ind2(:,1) = [y2; x2]; %setting first position of the flashlight
%camera 3
figure()
imshow(vidFrames3_2(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x3,y3] = ginput(1);
ind3(:,1) = [y3; x3]; %setting first position of the flashlight

close all; %close the cameras after finding the initial position

rw = 15; %size of row windows
cw = 15; %size of column windows
tr = 0; %temporary row size
tc = 0; %temporary column size
m1 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 1
m2 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 2
m3 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 3

% window for camera 1
for jj=2:size(vidFrames1_2,4)
    r1 = ind1(1,jj-1); %prior row location of flashlight
    c1 = ind1(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r1-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r1-1+(r1 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r1+rw) > size(vidFrames1_2,1) %if the window is larger than the size of the frame
        tr = size(vidFrames1_2,1)-r1+(r1 == size(vidFrames1_2,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c1-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c1-1+(c1 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c1+cw) > size(vidFrames1_2,1) %if the window is larger than the size of the frame
        tc = size(vidFrames1_2,1)-c1+(c1 == size(vidFrames1_2,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m1 = vidFrames1_2((r1-tr):(r1+tr),(c1-tc):(c1+tc),:,jj); %windowing the flashlight
    tm1 = (sum(m1,3) == max(max(sum(m1,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm1); %place the maximum intensity pixel in the (1,1) position
    ind1(1,jj) = r(1)+(r1-tr)-1; %extract the location of the pixel
    ind1(2,jj) = c(1)+(c1-tc)-1; %extract the location of the pixel
end
% window for camera 2
for jj=2:size(vidFrames2_2,4)
    r2 = ind2(1,jj-1); %prior row location of flashlight
    c2 = ind2(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r2-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r2-1+(r2 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r2+rw) > size(vidFrames2_2,1) %if the window is larger than the size of the frame
        tr = size(vidFrames2_2,1)-r2+(r2 == size(vidFrames2_2,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c2-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c2-1+(c2 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c2+cw) > size(vidFrames2_2,1) %if the window is larger than the size of the frame
        tc = size(vidFrames2_2,1)-c2+(c2 == size(vidFrames2_2,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m2 = vidFrames2_2((r2-tr):(r2+tr),(c2-tc):(c2+tc),:,jj); %windowing the flashlight
    tm2 = (sum(m2,3) == max(max(sum(m2,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm2); %place the maximum intensity pixel in the (1,1) position
    ind2(1,jj) = r(1)+(r2-tr)-1; %extract the location of the pixel
    ind2(2,jj) = c(1)+(c2-tc)-1; %extract the location of the pixel
end
% window for camera 3
for jj=2:size(vidFrames3_2,4)
    r3 = ind3(1,jj-1); %prior row location of flashlight
    c3 = ind3(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r3-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r3-1+(r3 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r3+rw) > size(vidFrames3_2,1) %if the window is larger than the size of the frame
        tr = size(vidFrames3_2,1)-r3+(r3 == size(vidFrames3_2,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c3-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c3-1+(c3 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c3+cw) > size(vidFrames3_2,1) %if the window is larger than the size of the frame
        tc = size(vidFrames3_2,1)-c3+(c3 == size(vidFrames3_2,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m3 = vidFrames3_2((r3-tr):(r3+tr),(c3-tc):(c3+tc),:,jj); %windowing the flashlight
    tm3 = (sum(m3,3) == max(max(sum(m3,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm3); %place the maximum intensity pixel in the (1,1) position
    ind3(1,jj) = r(1)+(r3-tr)-1; %extract the location of the pixel
    ind3(2,jj) = c(1)+(c3-tc)-1; %extract the location of the pixel
end

% aligning the 3 cameras to have max and min occur at the same time
for i = 1:2
    ind1(i,:) = (ind1(i,:)-min(ind1(i,:)))/max(ind1(i,:)); %camera 1
    ind2(i,:) = (ind2(i,:)-min(ind2(i,:)))/max(ind2(i,:)); %camera 2
    ind3(i,:) = (ind3(i,:)-min(ind3(i,:)))/max(ind3(i,:)); %camera 3
end

% verifying that the cameras are aligned
%figure()
%plot(ind1(1,2:end)), hold on
%plot(ind2(1,10:end))
%plot(ind3(2,1:end))
%xlabel('Frame'), ylabel('Row Index'), title('Normalized Indices over Time')
%legend('Camera 1', 'Camera 2', 'Camera 3')

% setting all 3 cameras to start at the same time
ind1 = ind1(:,12:end); %camera 1
ind2 = ind2(:,:); %camera 2
ind3 = ind3(:,12:end); %camera 3
t = min([size(ind1,2),size(ind2,2),size(ind3,2)]); %minimum number of time-steps
t0 = linspace(0,1,t); %time discretization
X = zeros(6,t); %cropping position matrix to be the same length for all cameras

% filling in the position matrix
X(1,:) = ind1(1,1:t);
X(2,:) = ind1(2,1:t);
X(3,:) = ind2(1,1:t);
X(4,:) = ind2(2,1:t);
X(5,:) = ind3(1,1:t);
X(6,:) = ind3(2,1:t);

X = X-repmat(mean(X,2),1,t); %subtract the mean
[u,s,v] = svd(X/sqrt(t-1)); %finding the SVD
sig = diag(s);
var = sig/sum(sig); %computing variance
Y = u.'*X; %projection of principal components

% PCA plots
figure()
subplot(3,2,1), plot(sig,'ko-') %singular values
xlabel('Modes'), ylabel('Sigma'), title('Singular Values') 
subplot(3,2,2), plot(var*100,'ko-') %percentage of variance
xlabel('Modes'), ylabel('Percentage'), title('Variance'), 
subplot(3,1,2) %projection of PCA
plot(t0,Y(1,:),'k',t0,Y(2,:),'r',t0,Y(3,:),'b',t0,Y(4,:),'c',t0,Y(5,:),'g',t0,Y(6,:),'m','Linewidth',[2])
legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Location','EastOutside')
xlabel('Time'), ylabel('Principal Components'), title('PCA Projection over Time')
subplot(3,1,3) %time behavior
plot(t0,v(1,:),'k',t0,v(2,:),'r',t0,v(3,:),'b',t0,v(4,:),'c',t0,v(5,:),'g',t0,v(6,:),'m')
legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Location','EastOutside')
xlabel('Time'), ylabel('PCA Data'), title('Time Evolution Behavior over PCA Modes')

%% Test 3: Horizontal Displacement (camN_3 where N = 1,2,3)
clear all; close all; clc

% loading cam files
load('cam1_3.mat'); %camera 1
load('cam2_3.mat'); %camera 2
load('cam3_3.mat'); %camera 3

% finding initial x,y coordinates from each video by clicking on the flashlight
ind1 = zeros(2,size(vidFrames1_3,4)); %camera 1
ind2 = zeros(2,size(vidFrames2_3,4)); %camera 1
ind3 = zeros(2,size(vidFrames3_3,4)); %camera 3
%camera 1
figure()
imshow(vidFrames1_3(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x1,y1] = ginput(1);
ind1(:,1) = [y1; x1]; %setting first position of the flashlight
%camera 2
figure()
imshow(vidFrames2_3(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x2,y2] = ginput(1);
ind2(:,1) = [y2; x2]; %setting first position of the flashlight
%camera 3
figure()
imshow(vidFrames3_3(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x3,y3] = ginput(1);
ind3(:,1) = [y3; x3]; %setting first position of the flashlight

close all; %close the cameras after finding the initial position

rw = 15; %size of row windows
cw = 15; %size of column windows
tr = 0; %temporary row size
tc = 0; %temporary column size
m1 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 1
m2 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 2
m3 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 3

% window for camera 1
for jj=2:size(vidFrames1_3,4)
    r1 = ind1(1,jj-1); %prior row location of flashlight
    c1 = ind1(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r1-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r1-1+(r1 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r1+rw) > size(vidFrames1_3,1) %if the window is larger than the size of the frame
        tr = size(vidFrames1_3,1)-r1+(r1 == size(vidFrames1_3,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c1-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c1-1+(c1 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c1+cw) > size(vidFrames1_3,1) %if the window is larger than the size of the frame
        tc = size(vidFrames1_3,1)-c1+(c1 == size(vidFrames1_3,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m1 = vidFrames1_3((r1-tr):(r1+tr),(c1-tc):(c1+tc),:,jj); %windowing the flashlight
    tm1 = (sum(m1,3) == max(max(sum(m1,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm1); %place the maximum intensity pixel in the (1,1) position
    ind1(1,jj) = r(1)+(r1-tr)-1; %extract the location of the pixel
    ind1(2,jj) = c(1)+(c1-tc)-1; %extract the location of the pixel
end
% window for camera 2
for jj=2:size(vidFrames2_3,4)
    r2 = ind2(1,jj-1); %prior row location of flashlight
    c2 = ind2(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r2-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r2-1+(r2 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r2+rw) > size(vidFrames2_3,1) %if the window is larger than the size of the frame
        tr = size(vidFrames2_3,1)-r2+(r2 == size(vidFrames2_3,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c2-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c2-1+(c2 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c2+cw) > size(vidFrames2_3,1) %if the window is larger than the size of the frame
        tc = size(vidFrames2_3,1)-c2+(c2 == size(vidFrames2_3,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m2 = vidFrames2_3((r2-tr):(r2+tr),(c2-tc):(c2+tc),:,jj); %windowing the flashlight
    tm2 = (sum(m2,3) == max(max(sum(m2,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm2); %place the maximum intensity pixel in the (1,1) position
    ind2(1,jj) = r(1)+(r2-tr)-1; %extract the location of the pixel
    ind2(2,jj) = c(1)+(c2-tc)-1; %extract the location of the pixel
end
% window for camera 3
for jj=2:size(vidFrames3_3,4)
    r3 = ind3(1,jj-1); %prior row location of flashlight
    c3 = ind3(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r3-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r3-1+(r3 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r3+rw) > size(vidFrames3_3,1) %if the window is larger than the size of the frame
        tr = size(vidFrames3_3,1)-r3+(r3 == size(vidFrames3_3,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c3-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c3-1+(c3 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c3+cw) > size(vidFrames3_3,1) %if the window is larger than the size of the frame
        tc = size(vidFrames3_3,1)-c3+(c3 == size(vidFrames3_3,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m3 = vidFrames3_3((r3-tr):(r3+tr),(c3-tc):(c3+tc),:,jj); %windowing the flashlight
    tm3 = (sum(m3,3) == max(max(sum(m3,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm3); %place the maximum intensity pixel in the (1,1) position
    ind3(1,jj) = r(1)+(r3-tr)-1; %extract the location of the pixel
    ind3(2,jj) = c(1)+(c3-tc)-1; %extract the location of the pixel
end

% aligning the 3 cameras to have max and min occur at the same time
for i = 1:2
    ind1(i,:) = (ind1(i,:)-min(ind1(i,:)))/max(ind1(i,:)); %camera 1
    ind2(i,:) = (ind2(i,:)-min(ind2(i,:)))/max(ind2(i,:)); %camera 2
    ind3(i,:) = (ind3(i,:)-min(ind3(i,:)))/max(ind3(i,:)); %camera 3
end

% verifying that the cameras are aligned
%figure()
%plot(ind1(1,2:end)), hold on
%plot(ind2(1,10:end))
%plot(ind3(2,1:end))
%xlabel('Frame'), ylabel('Row Index'), title('Normalized Indices over Time')
%legend('Camera 1', 'Camera 2', 'Camera 3')

% setting all 3 cameras to start at the same time
ind1 = ind1(:,12:end); %camera 1
ind2 = ind2(:,1:end); %camera 2
ind3 = ind3(:,4:end); %camera 3
t = min([size(ind1,2),size(ind2,2),size(ind3,2)]); %minimum number of time-steps
t0 = linspace(0,1,t); %time discretization
X = zeros(6,t); %cropping position matrix to be the same length for all cameras

% filling in the position matrix
X(1,:) = ind1(1,1:t);
X(2,:) = ind1(2,1:t);
X(3,:) = ind2(1,1:t);
X(4,:) = ind2(2,1:t);
X(5,:) = ind3(1,1:t);
X(6,:) = ind3(2,1:t);

X = X-repmat(mean(X,2),1,t); %subtract the mean
[u,s,v] = svd(X/sqrt(t-1)); %finding the SVD
sig = diag(s);
var = sig/sum(sig); %computing variance
Y = u.'*X; %projection of principal components

% PCA plots
figure()
subplot(3,2,1), plot(sig,'ko-') %singular values
xlabel('Modes'), ylabel('Sigma'), title('Singular Values') 
subplot(3,2,2), plot(var*100,'ko-') %percentage of variance
xlabel('Modes'), ylabel('Percentage'), title('Variance'), 
subplot(3,1,2) %projection of PCA
plot(t0,Y(1,:),'k',t0,Y(2,:),'r',t0,Y(3,:),'b',t0,Y(4,:),'c',t0,Y(5,:),'g',t0,Y(6,:),'m','Linewidth',[2])
legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Location','EastOutside')
xlabel('Time'), ylabel('Principal Components'), title('PCA Projection over Time')
subplot(3,1,3) %time behavior
plot(t0,v(1,:),'k',t0,v(2,:),'r',t0,v(3,:),'b',t0,v(4,:),'c',t0,v(5,:),'g',t0,v(6,:),'m')
legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Location','EastOutside')
xlabel('Time'), ylabel('PCA Data'), title('Time Evolution Behavior over PCA Modes')

%% Test 4: Horizontal Displacement and Rotation (camN_4 where N = 1,2,3)
clear all; close all; clc

% loading cam files
load('cam1_4.mat'); %camera 1
load('cam2_4.mat'); %camera 2
load('cam3_4.mat'); %camera 3

% finding initial x,y coordinates from each video by clicking on the flashlight
ind1 = zeros(2,size(vidFrames1_4,4)); %camera 1
ind2 = zeros(2,size(vidFrames2_4,4)); %camera 1
ind3 = zeros(2,size(vidFrames3_4,4)); %camera 3
%camera 1
figure()
imshow(vidFrames1_4(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x1,y1] = ginput(1);
ind1(:,1) = [y1; x1]; %setting first position of the flashlight
%camera 2
figure()
imshow(vidFrames2_4(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x2,y2] = ginput(1);
ind2(:,1) = [y2; x2]; %setting first position of the flashlight
%camera 3
figure()
imshow(vidFrames3_4(:,:,:,1)), title('Click the Flashlight on top of the Paint Can')
[x3,y3] = ginput(1);
ind3(:,1) = [y3; x3]; %setting first position of the flashlight

close all; %close the cameras after finding the initial position

rw = 15; %size of row windows
cw = 15; %size of column windows
tr = 0; %temporary row size
tc = 0; %temporary column size
m1 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 1
m2 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 2
m3 = zeros(rw+1,cw+1,3,'uint8'); %window for camera 3

% window for camera 1
for jj=2:size(vidFrames1_4,4)
    r1 = ind1(1,jj-1); %prior row location of flashlight
    c1 = ind1(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r1-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r1-1+(r1 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r1+rw) > size(vidFrames1_4,1) %if the window is larger than the size of the frame
        tr = size(vidFrames1_4,1)-r1+(r1 == size(vidFrames1_4,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c1-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c1-1+(c1 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c1+cw) > size(vidFrames1_4,1) %if the window is larger than the size of the frame
        tc = size(vidFrames1_4,1)-c1+(c1 == size(vidFrames1_4,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m1 = vidFrames1_4((r1-tr):(r1+tr),(c1-tc):(c1+tc),:,jj); %windowing the flashlight
    tm1 = (sum(m1,3) == max(max(sum(m1,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm1); %place the maximum intensity pixel in the (1,1) position
    ind1(1,jj) = r(1)+(r1-tr)-1; %extract the location of the pixel
    ind1(2,jj) = c(1)+(c1-tc)-1; %extract the location of the pixel
end
% window for camera 2
for jj=2:size(vidFrames2_4,4)
    r2 = ind2(1,jj-1); %prior row location of flashlight
    c2 = ind2(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r2-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r2-1+(r2 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r2+rw) > size(vidFrames2_4,1) %if the window is larger than the size of the frame
        tr = size(vidFrames2_4,1)-r2+(r2 == size(vidFrames2_4,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c2-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c2-1+(c2 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c2+cw) > size(vidFrames2_4,1) %if the window is larger than the size of the frame
        tc = size(vidFrames2_4,1)-c2+(c2 == size(vidFrames2_4,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m2 = vidFrames2_4((r2-tr):(r2+tr),(c2-tc):(c2+tc),:,jj); %windowing the flashlight
    tm2 = (sum(m2,3) == max(max(sum(m2,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm2); %place the maximum intensity pixel in the (1,1) position
    ind2(1,jj) = r(1)+(r2-tr)-1; %extract the location of the pixel
    ind2(2,jj) = c(1)+(c2-tc)-1; %extract the location of the pixel
end
% window for camera 3
for jj=2:size(vidFrames3_4,4)
    r3 = ind3(1,jj-1); %prior row location of flashlight
    c3 = ind3(2,jj-1); %prior column location of flashlight
    
    % ensure the window stays in frame
    if (r3-rw) < 1 %if the previous row is smaller than size of the row windows
        tr = r3-1+(r3 == 1); %adjust the temporary row size to keep the window in frame
    elseif (r3+rw) > size(vidFrames3_4,1) %if the window is larger than the size of the frame
        tr = size(vidFrames3_4,1)-r3+(r3 == size(vidFrames3_4,1)); %adjust the temporary row size to keep the window in frame
    else
        tr = rw; %otherwise, keep the temporary row size equal to the size of the row windows
    end
    if (c3-cw) < 1 %if the previous column is smaller than size of the column windows
        tc = c3-1+(c3 == 1); %adjust the temporary column size to keep the window in frame
    elseif (c3+cw) > size(vidFrames3_4,1) %if the window is larger than the size of the frame
        tc = size(vidFrames3_4,1)-c3+(c3 == size(vidFrames3_4,1)); %adjust the temporary column size to keep the window in frame
    else
        tc = cw; %otherwise, keep the temporary column size equal to the size of the column windows
    end
    
    m3 = vidFrames3_4((r3-tr):(r3+tr),(c3-tc):(c3+tc),:,jj); %windowing the flashlight
    tm3 = (sum(m3,3) == max(max(sum(m3,3)))); %finding the maximum intensity pixel
    [r,c] = find(tm3); %place the maximum intensity pixel in the (1,1) position
    ind3(1,jj) = r(1)+(r3-tr)-1; %extract the location of the pixel
    ind3(2,jj) = c(1)+(c3-tc)-1; %extract the location of the pixel
end

% aligning the 3 cameras to have max and min occur at the same time
for i = 1:2
    ind1(i,:) = (ind1(i,:)-min(ind1(i,:)))/max(ind1(i,:)); %camera 1
    ind2(i,:) = (ind2(i,:)-min(ind2(i,:)))/max(ind2(i,:)); %camera 2
    ind3(i,:) = (ind3(i,:)-min(ind3(i,:)))/max(ind3(i,:)); %camera 3
end

% verifying that the cameras are aligned
%figure()
%plot(ind1(1,2:end)), hold on
%plot(ind2(1,10:end))
%plot(ind3(2,1:end))
%xlabel('Frame'), ylabel('Row Index'), title('Normalized Indices over Time')
%legend('Camera 1', 'Camera 2', 'Camera 3')

% setting all 3 cameras to start at the same time
ind1 = ind1(:,2:end); %camera 1
ind2 = ind2(:,10:end); %camera 2
ind3 = ind3(:,1:end); %camera 3
t = min([size(ind1,2),size(ind2,2),size(ind3,2)]); %minimum number of time-steps
t0 = linspace(0,1,t); %time discretization
X = zeros(6,t); %cropping position matrix to be the same length for all cameras

% filling in the position matrix
X(1,:) = ind1(1,1:t);
X(2,:) = ind1(2,1:t);
X(3,:) = ind2(1,1:t);
X(4,:) = ind2(2,1:t);
X(5,:) = ind3(1,1:t);
X(6,:) = ind3(2,1:t);

X = X-repmat(mean(X,2),1,t); %subtract the mean
[u,s,v] = svd(X/sqrt(t-1)); %finding the SVD
sig = diag(s);
var = sig/sum(sig); %computing variance
Y = u.'*X; %projection of principal components

% PCA plots
figure()
subplot(3,2,1), plot(sig,'ko-') %singular values
xlabel('Modes'), ylabel('Sigma'), title('Singular Values') 
subplot(3,2,2), plot(var*100,'ko-') %percentage of variance
xlabel('Modes'), ylabel('Percentage'), title('Variance'), 
subplot(3,1,2) %projection of PCA
plot(t0,Y(1,:),'k',t0,Y(2,:),'r',t0,Y(3,:),'b',t0,Y(4,:),'c',t0,Y(5,:),'g',t0,Y(6,:),'m','Linewidth',[2])
legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Location','EastOutside')
xlabel('Time'), ylabel('Principal Components'), title('PCA Projection over Time')
subplot(3,1,3) %time behavior
plot(t0,v(1,:),'k',t0,v(2,:),'r',t0,v(3,:),'b',t0,v(4,:),'c',t0,v(5,:),'g',t0,v(6,:),'m')
legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Location','EastOutside')
xlabel('Time'), ylabel('PCA Data'), title('Time Evolution Behavior over PCA Modes')