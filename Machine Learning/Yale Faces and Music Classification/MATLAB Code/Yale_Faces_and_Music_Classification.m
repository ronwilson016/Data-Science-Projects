%% Part 1.1: Yale Faces B (Cropped Images)
clear all; close all; clc

% loading files
crop = dir('CroppedYale'); %cropped faces
crop = crop(3:end); %excluding the . and .. entries

% cropped faces
c_data = []; %all images where each column represents a new image
for i = 1:length(crop)
    % getting the sub-directories
    name = crop(i).name;
    k = ['CroppedYale','/',name];
    sub_crop = dir(k);
    cd(k) %changing directory to the subdirectories to read images
    sub_crop = sub_crop(3:end); %excluding the . and .. entries
    
    for j = 1:length(sub_crop)
        faces = imread(sub_crop(j).name); %each image
        faces = reshape(faces,192*168,1); %reshaping image into a column vector
        c_data = [c_data faces]; %adding each column vector to the data matrix
    end
    cd ..\.. %return to the original directory
end

% computing SVD
c_data = double(c_data);
[u,s,v] = svd(c_data,'econ'); %reduced svd of cropped images
sig = diag(s); var = sig/sum(sig); %computing variance

% plotting singular values
figure(1)
plot(var*100,'go','LineWidth',[1.5])
xlabel('Cropped Faces'), ylabel('% of Singular Values')
title('Singular Value Spectrum (Cropped Images)')

% image reconstruction
r = [4; 50; 200]; %setting different singular values
c_reconstruct1 = u*s(:,1:r(1))*v(:,1:r(1))'; %reconstructed images
c_reconstruct2 = u*s(:,1:r(2))*v(:,1:r(2))'; %reconstructed images
c_reconstruct3 = u*s(:,1:r(3))*v(:,1:r(3))'; %reconstructed images

figure(2)
n = [75, 503, 1000, 2432]; %choosing random images
for jj = 1:length(n)
    subplot(4,2,2*jj-1)
    imshow(uint8(reshape(c_data(:,n(jj)),192,168))); %viewing original image
    title('Original Image')
    
    subplot(4,2,2*jj)
    imshow(uint8(reshape(c_reconstruct1(:,n(jj)),192,168))); %viewing reconstructed image
    title('Reconstructed Image')
end

% testing number of modes for accurate image reconstruction
figure(3)
subplot(3,2,1)
imshow(uint8(reshape(c_data(:,n(1)),192,168)));
title('Original Image')
subplot(3,2,2)
imshow(uint8(reshape(c_reconstruct1(:,n(1)),192,168)));
title('Reconstructed Image (r = 4)')
subplot(3,2,3)
imshow(uint8(reshape(c_data(:,n(1)),192,168)));
title('Original Image')
subplot(3,2,4)
imshow(uint8(reshape(c_reconstruct2(:,n(1)),192,168)));
title('Reconstructed Image (r = 50)')
subplot(3,2,5)
imshow(uint8(reshape(c_data(:,n(1)),192,168)));
title('Original Image')
subplot(3,2,6)
imshow(uint8(reshape(c_reconstruct3(:,n(1)),192,168)));
title('Reconstructed Image (r = 200)')

%% Part 1.2: Yale Faces B (Original Images)
clear all; close all; clc

% loading files
uncrop = dir('yalefaces'); %original images
uncrop = uncrop(3:end); %excluding the . and .. entries

% uncropped faces
u_data = []; %all images where each column represents a new image
cd('yalefaces') %changing directory to read images
for i = 1:length(uncrop)
    faces = imread(uncrop(i).name); %each image
    faces = reshape(faces,243*320,1); %reshaping image into a column vector
    u_data = [u_data faces]; %adding each column vector to the data matrix
end
cd .. %return to the original directory

% computing SVD
u_data = double(u_data);
[u,s,v] = svd(u_data,'econ'); %reduced svd of uncropped images
sig = diag(s); var = sig/sum(sig); %computing variance

% plotting singular values
figure(1)
plot(var*100,'go','LineWidth',[1.5])
xlabel('Uncropped Faces'), ylabel('% of Singular Values')
title('Singular Value Spectrum (Original Images)')

% image reconstruction
r = [6; 30; 150]; %setting different singular values
u_reconstruct1 = u*s(:,1:r(1))*v(:,1:r(1))'; %reconstructed images
u_reconstruct2 = u*s(:,1:r(2))*v(:,1:r(2))'; %reconstructed images
u_reconstruct3 = u*s(:,1:r(3))*v(:,1:r(3))'; %reconstructed images

figure(2)
n = [17, 53, 100, 165]; %choosing random images
for jj = 1:length(n)
    subplot(4,2,2*jj-1)
    imshow(uint8(reshape(u_data(:,n(jj)),243,320))); %viewing original image
    title('Original Image')
    
    subplot(4,2,2*jj)
    imshow(uint8(reshape(u_reconstruct1(:,n(jj)),243,320))); %viewing reconstructed image
    title('Reconstructed Image')
end

% testing number of modes for accurate image reconstruction
figure(3)
subplot(3,2,1)
imshow(uint8(reshape(u_data(:,n(1)),243,320)));
title('Original Image')
subplot(3,2,2)
imshow(uint8(reshape(u_reconstruct1(:,n(1)),243,320)));
title('Reconstructed Image (r = 6)')
subplot(3,2,3)
imshow(uint8(reshape(u_data(:,n(1)),243,320)));
title('Original Image')
subplot(3,2,4)
imshow(uint8(reshape(u_reconstruct2(:,n(1)),243,320)));
title('Reconstructed Image (r = 30)')
subplot(3,2,5)
imshow(uint8(reshape(u_data(:,n(1)),243,320)));
title('Original Image')
subplot(3,2,6)
imshow(uint8(reshape(u_reconstruct3(:,n(1)),243,320)));
title('Reconstructed Image (r = 150)')

%% Part 2.1: Test 1 (Band Classification)
clear all; close all; clc

% loading data and storing it in a matrix
test_1 = dir('Test 1');
test_1 = test_1(3:end); %excluding the . and .. entries

features = 17; %number of features
y = ones(1,features); %vector containing features for sampling
Y = [y 2*y 3*y]; %creating feature space for testing
y0 = Y; %feature space

n = 220500; %number of samples
y_0 = zeros(n,1); %vector containing each sample

for i=1:length(test_1)
    name = test_1(i).name;
    k = ['Test 1','/',name];
    band = dir(k);
    cd(k) %changing directory to the subdirectories to read audio files
    band = band(3:end);
    
    for j = 1:length(band)
        [y,Fs] = audioread(band(j).name); %reading audio files
        y = mean(y,2); %taking the mean of the two columns in each file
        if length(y(:,1))>n %if the length of the mean is larger than the number of samples
            y = y(1:n,:); %store the data in the second vector
        else %otherwise
            n = length(y(:,1)); %change the number of samples to the size of the data
            y_0 = y_0(1:n,:); %store the data in the first vector
        end
        y_0 = [y_0 y]; %combine the first and second vectors
    end
    cd ..\.. %return to the original directory
end
y_0 = y_0(:,2:end); %removing the column of zeros
test0 = y_0; %copying the data for testing

% randomly creating training and testing sets
ind = randi([1 length(Y)],1,10); %random indices
test = test0(:,ind);
y0_test = Y(:,ind); %testing set
train = test0;
y0_train = y0; %training set

% generating spectrogram of the training set
s_train = []; %training set
for i = 1:length(train(1,:))
    f_train = fft(train(:,i)); %shifting into the frequency domain
    spec = abs(fftshift(f_train)); %storing frequencies in the spectrogram
    s_train = [s_train spec]; %generating a matrix of the spectrogram data
end
[~,b] = size(s_train); %size of the data
s_train = s_train-repmat(mean(s_train,2),1,b); %subtracting the mean
[~,~,v] = svd(s_train','econ'); %reduced SVD of the training set spectrogram
v = v(:,1:length(Y)); %taking the feature of v

% classification using KNN
m1 = fitcknn(v',y0_train,'NumNeighbors',5); %model from knn
l1 = predict(m1,test'); %classification labels
r1 = 0; %number of accurate tests
for i = 1:length(l1)
    if l1(i) == y0_test(i) %if the classification matches the test data
        r1 = r1+1; %counting the number of correct tests
    end
end
q1 = r1/length(l1); %accuracy of the classification
%disp(['Accuracy of Test 1 Using KNN is ', num2str(q1)])

% classification using SVM
m2 = fitcecoc(v',y0_train); %model from svm
l2 = predict(m2,test'); %classification labels
r2 = 0; %number of accurate tests
for i = 1:length(l2)
    if l2(i) == y0_test(i) %if the classification matches the test data
        r2 = r2+1; %counting the number of correct tests
    end
end
q2 = r2/length(l2); %accuracy of the classification
%disp(['Accuracy of Test 1 Using SVM is ', num2str(q2)])

% plotting accuracies from two different classifications
knn_acc = q1*100; %percentage of KNN accuracy
svm_acc = q2*100; %percentage of SVM accuracy
acc_x = categorical({'KNN', 'SVM'}); %x-axis of bar graph
acc_y = [knn_acc svm_acc]; %y-axis of bar graph
bar(acc_x, acc_y, 'g')
xlabel('Classification Methods'), ylabel('Accuracy (%)')
title('Accuracy of Test 1')

%% Part 2.2: Test 2 (The Case for Seattle)
clear all; close all; clc

% loading data and storing it in a matrix
test_2 = dir('Test 2');
test_2 = test_2(3:end); %excluding the . and .. entries

features = 17; %number of features
y = ones(1,features); %vector containing features for sampling
Y = [y 2*y 3*y]; %creating feature space for testing
y0 = Y; %feature space

n = 220500; %number of samples
y_0 = zeros(n,1); %vector containing each sample

for i=1:length(test_2)
    name = test_2(i).name;
    k = ['Test 2','/',name];
    band = dir(k);
    cd(k) %changing directory to the subdirectories to read audio files
    band = band(3:end);
    
    for j = 1:length(band)
        [y,Fs] = audioread(band(j).name); %reading audio files
        y = mean(y,2); %taking the mean of the two columns in each file
        if length(y(:,1))>n %if the length of the mean is larger than the number of samples
            y = y(1:n,:); %store the data in the second vector
        else %otherwise
            n = length(y(:,1)); %change the number of samples to the size of the data
            y_0 = y_0(1:n,:); %store the data in the first vector
        end
        y_0 = [y_0 y]; %combine the first and second vectors
    end
    cd ..\.. %return to the original directory
end
y_0 = y_0(:,2:end); %removing the column of zeros
test0 = y_0; %copying the data for testing

% randomly creating training and testing sets
ind = randi([1 length(Y)],1,10); %random indices
test = test0(:,ind);
y0_test = Y(:,ind); %testing set
train = test0;
y0_train = y0; %training set

% generating spectrogram of the training set
s_train = []; %training set
for i = 1:length(train(1,:))
    f_train = fft(train(:,i)); %shifting into the frequency domain
    spec = abs(fftshift(f_train)); %storing frequencies in the spectrogram
    s_train = [s_train spec]; %generating a matrix of the spectrogram data
end
[~,b] = size(s_train); %size of the data
s_train = s_train-repmat(mean(s_train,2),1,b); %subtracting the mean
[~,~,v] = svd(s_train','econ'); %reduced SVD of the training set spectrogram
v = v(:,1:length(Y)); %taking the feature of v

% classification using KNN
m1 = fitcknn(v',y0_train,'NumNeighbors',5); %model from knn
l1 = predict(m1,test'); %classification labels
r1 = 0; %number of accurate tests
for i = 1:length(l1)
    if l1(i) == y0_test(i) %if the classification matches the test data
        r1 = r1+1; %counting the number of correct tests
    end
end
q1 = r1/length(l1); %accuracy of the classification
%disp(['Accuracy of Test 2 Using KNN is ', num2str(q1)])

% classification using SVM
m2 = fitcecoc(v',y0_train); %model from svm
l2 = predict(m2,test'); %classification labels
r2 = 0; %number of accurate tests
for i = 1:length(l2)
    if l2(i) == y0_test(i) %if the classification matches the test data
        r2 = r2+1; %counting the number of correct tests
    end
end
q2 = r2/length(l2); %accuracy of the classification
%disp(['Accuracy of Test 2 Using SVM is ', num2str(q2)])

% plotting accuracies from two different classifications
knn_acc = q1*100; %percentage of KNN accuracy
svm_acc = q2*100; %percentage of SVM accuracy
acc_x = categorical({'KNN', 'SVM'}); %x-axis of bar graph
acc_y = [knn_acc svm_acc]; %y-axis of bar graph
bar(acc_x, acc_y, 'g')
xlabel('Classification Methods'), ylabel('Accuracy (%)')
title('Accuracy of Test 2')

%% Part 2.3: Test 3 (Genre Classification)
clear all; close all; clc

% loading data and storing it in a matrix
test_3 = dir('Test 3');
test_3 = test_3(3:end); %excluding the . and .. entries

features = 53; %number of features
y = ones(1,features); %vector containing features for sampling
Y = [y 2*y 3*y]; %creating feature space for testing
y0 = Y; %feature space

n = 220500; %number of samples
y_0 = zeros(n,1); %vector containing each sample

for i=1:length(test_3)
    name = test_3(i).name;
    k = ['Test 3','/',name];
    band = dir(k);
    cd(k) %changing directory to the subdirectories to read audio files
    band = band(3:end);
    
    for j = 1:length(band)
        [y,Fs] = audioread(band(j).name); %reading audio files
        y = mean(y,2); %taking the mean of the two columns in each file
        if length(y(:,1))>n %if the length of the mean is larger than the number of samples
            y = y(1:n,:); %store the data in the second vector
        else %otherwise
            n = length(y(:,1)); %change the number of samples to the size of the data
            y_0 = y_0(1:n,:); %store the data in the first vector
        end
        y_0 = [y_0 y]; %combine the first and second vectors
    end
    cd ..\.. %return to the original directory
end
y_0 = y_0(:,2:end); %removing the column of zeros
test0 = y_0; %copying the data for testing

% randomly creating training and testing sets
ind = randi([1 length(Y)],1,10); %random indices
test = test0(:,ind);
y0_test = Y(:,ind); %testing set
train = test0;
y0_train = y0; %training set

% generating spectrogram of the training set
s_train = []; %training set
for i = 1:length(train(1,:))
    f_train = fft(train(:,i)); %shifting into the frequency domain
    spec = abs(fftshift(f_train)); %storing frequencies in the spectrogram
    s_train = [s_train spec]; %generating a matrix of the spectrogram data
end
[~,b] = size(s_train); %size of the data
s_train = s_train-repmat(mean(s_train,2),1,b); %subtracting the mean
[~,~,v] = svd(s_train','econ'); %reduced SVD of the training set spectrogram
v = v(:,1:length(Y)); %taking the feature of v

% classification using KNN
m1 = fitcknn(v',y0_train,'NumNeighbors',5); %model from knn
l1 = predict(m1,test'); %classification labels
r1 = 0; %number of accurate tests
for i = 1:length(l1)
    if l1(i) == y0_test(i) %if the classification matches the test data
        r1 = r1+1; %counting the number of correct tests
    end
end
q1 = r1/length(l1); %accuracy of the classification
%disp(['Accuracy of Test 3 Using KNN is ', num2str(q1)])

% classification using SVM
m2 = fitcecoc(v',y0_train); %model from svm
l2 = predict(m2,test'); %classification labels
r2 = 0; %number of accurate tests
for i = 1:length(l2)
    if l2(i) == y0_test(i) %if the classification matches the test data
        r2 = r2+1; %counting the number of correct tests
    end
end
q2 = r2/length(l2); %accuracy of the classification
%disp(['Accuracy of Test 3 Using SVM is ', num2str(q2)])

% plotting accuracies from two different classifications
knn_acc = q1*100; %percentage of KNN accuracy
svm_acc = q2*100; %percentage of SVM accuracy
acc_x = categorical({'KNN', 'SVM'}); %x-axis of bar graph
acc_y = [knn_acc svm_acc]; %y-axis of bar graph
bar(acc_x, acc_y, 'g')
xlabel('Classification Methods'), ylabel('Accuracy (%)')
title('Accuracy of Test 3')