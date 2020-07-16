clear all; close all; clc

%% Part 1.1: Yale Faces B (Cropped Images)

% loading files
crop = dir('CroppedYale'); % cropped faces
crop = crop(3:end); % excluding the . and .. entries

% storing images into column vectors
c_data = []; % pre-allocating cropped images data matrix
for i = 1:length(crop)
    % getting the sub-directories
    name = crop(i).name;
    k = ['CroppedYale','/',name];
    sub_crop = dir(k);
    cd(k) % changing directory to the subdirectories to read images
    sub_crop = sub_crop(3:end); % excluding the . and .. entries
    
    for j = 1:length(sub_crop)
        faces = imread(sub_crop(j).name); % each image
        faces = reshape(faces,192*168,1); % reshaping image into a column vector
        c_data = [c_data faces]; % adding each column vector to the data matrix
    end
    cd ..\.. % return to the original directory
end

% computing SVD
c_data = double(c_data); % converting data matrix into double for SVD
[m,n] = size(c_data); % size of the cropped data set
[uc,sc,vc] = svd(c_data,'econ'); % reduced svd of cropped images
sig_c = diag(sc); var_c = sig_c/sum(sig_c); % computing the variance

% plotting singular values to determine rank r
figure(1), plot(var_c*100,'ro','Linewidth',[2]), xlabel('Singular Values')
ylabel('% of Variance'), title('Singular Value Spectrum (Cropped Images)')

% image reconstruction
%rc = 4; % rank of the face space
rc = 1586; % rank of the face space (sum of var from 1 to 1586 equals 95%)
c_reconstruct = uc*sc(:,1:rc)*vc(:,1:rc)'; % reconstructed images

% plotting original images vs reconstructed images
figure(2)
nc = [70, 513, 1070, 2430]; % selecting four cropped images
for jj = 1:length(nc)
    subplot(4,2,2*jj-1)
    imshow(uint8(reshape(c_data(:,nc(jj)),192,168))); % viewing original image
    title('Original Image')
    
    subplot(4,2,2*jj)
    imshow(uint8(reshape(c_reconstruct(:,nc(jj)),192,168))); % viewing reconstructed image
    title('Reconstructed Image')
end

%% Part 1.2: Yale Faces B (Uncropped Images)

% loading files
uncrop = dir('yalefaces'); % uncropped images
uncrop = uncrop(3:end); % excluding the . and .. entries

% storing images into column vectors
u_data = []; % pre-allocating uncropped images data matrix
cd('yalefaces') % changing directory to read images
for i = 1:length(uncrop)
    faces = imread(uncrop(i).name); % each image
    faces = reshape(faces,243*320,1); % reshaping image into a column vector
    u_data = [u_data faces]; % adding each column vector to the data matrix
end
cd .. % return to the original directory

% computing SVD
u_data = double(u_data); % converting data matrix into double for SVD
[uu,su,vu] = svd(u_data,'econ'); % reduced svd of uncropped images
sig_u = diag(su); var_u = sig_u/sum(sig_u); % computing the variance

% plotting singular values
figure(3), plot(var_u*100,'ro','Linewidth',[2]), xlabel('Singular Values')
ylabel('% of Variance'), title('Singular Value Spectrum (Uncropped Images)')

% image reconstruction
%ru = 2; % rank of the face space
ru = 118; % rank of the face space (sum of var from 1 to 118 equals 95%)
u_reconstruct = uu*su(:,1:ru)*vu(:,1:ru)'; % reconstructed images

% plotting original images vs reconstructed images
figure(4)
nu = [13, 51, 109, 160]; % selecting four uncropped images
for jj = 1:length(nu)
    subplot(4,2,2*jj-1)
    imshow(uint8(reshape(u_data(:,nu(jj)),243,320))); % viewing original image
    title('Original Image')
    
    subplot(4,2,2*jj)
    imshow(uint8(reshape(u_reconstruct(:,nu(jj)),243,320))); % viewing reconstructed image
    title('Reconstructed Image')
end

%% Part 2.1: Face Identification (Test 1 - Face Classification)

% generating labels for the 38 individuals
labels1 = zeros(1,n); % pre-allocating a labeling set for individuals
for j = 1:38
    labels1(1+64*(j-1):64*j) = j;
end

L = 2000; % number of entries in the training set
q1 = randperm(n); % generating a random matrix for indexing
x1 = vc(1:n,1:rc); % taking the 95% accurate modes from the signature matrix

% splitting the data into training & testing sets
xtrain1 = x1(q1(1:L),:); % training set
xtest1 = x1(q1(L+1:end),:); % testing set

% getting the correct labels for the training & testing sets
ctrain1 = labels1(q1(1:L)); % labels for training
check1 = labels1(q1(L+1:end)); % labels for testing

% finding the classification labels using Support Vector Machines (SVM)
svm1 = fitcecoc(xtrain1,ctrain1); % SVM of training data & labels
pre1 = predict(svm1,xtest1); % applying SVM to testing data to get classification

% determining the accuracy of the classification
count1 = 0; % pre-setting a total count for accurate predictions
for i = 1:length(pre1)
    if pre1(i) == check1(i) % if the prediction was accurate
        count1 = count1 + 1; % add 1 to the count of accurate predictions
    end
end
accuracy1 = count1/length(pre1); % dividing the count by the total number of predictions

% plotting the accuracy of the classification method on individual faces
figure(5)
subplot(2,2,1:2), bar(pre1,'b'), title('Predictors vs True Labels'), hold on
plot(check1,'ro','Linewidth',[2]), xlabel('Testing Data'), ylabel('Labels')
legend({'Predictors','Labels'},'Location','EastOutside')
subplot(2,2,3), bar(pre1 == check1.','b','Linewidth',[2]), ylabel('Accuracy')
xlabel('Testing Data'), title('Accuracy of Predictors (1 = Correct, 0 = Incorrect)')
subplot(2,2,4), bar(accuracy1,'b'), ylabel('Accuracy'), set(gca,'xticklabel',{'SVM'})
title('Total Accuracy of SVM Classification on Individual Faces')

%% Part 2.2: Face Identification (Test 2 - Gender Classification)

% generating labels for the genders
labels2 = zeros(1,n); % pre-allocating a labeling set for genders
% female are folders 5, 15, 22, 27, 28, 32, 34, & 37
% However, folder 14 is missing, so for the ones above 14, subtract 1
for j = 1:38
    if j == 5 || j == 14 || j == 21 || j == 26 || j == 27 || j == 31 || j == 33 || j == 36
        labels2(1+64*(j-1):64*j) = 1; % female
    else
        labels2(1+64*(j-1):64*j) = 2; % male
    end
end

L = 2000; % number of entries in the training set
q2 = randperm(n); % generating a random matrix for indexing
x2 = vc(1:n,1:rc); % taking the 95% accurate modes from the signature matrix

% splitting the data into training & testing sets
xtrain2 = x2(q2(1:L),:); % training set
xtest2 = x2(q2(L+1:end),:); % testing set

% getting the correct labels for the training & testing sets
ctrain2 = labels2(q2(1:L)); % labels for training
check2 = labels2(q2(L+1:end)); % labels for testing

% finding the classification labels using Support Vector Machines (SVM)
svm2 = fitcsvm(xtrain2,ctrain2); % SVM of training data & labels
pre2 = predict(svm2,xtest2); % applying SVM to testing data to get classification

% determining the accuracy of the classification
count2 = 0; % pre-setting a total count for accurate predictions
for i = 1:length(pre2)
    if pre2(i) == check2(i) % if the prediction was accurate
        count2 = count2 + 1; % add 1 to the count of accurate predictions
    end
end
accuracy2 = count2/length(pre2); % dividing the count by the total number of predictions

% plotting the accuracy of the classification method on genders
figure(6)
subplot(2,2,1:2), bar(pre2,'b'), title('Predictors vs True Labels'), hold on
plot(check2,'ro','Linewidth',[2]), xlabel('Testing Data'), ylabel('Labels')
legend({'Predictors','Labels'},'Location','EastOutside')
subplot(2,2,3), bar(pre2 == check2.','b','Linewidth',[2]), ylabel('Accuracy')
xlabel('Testing Data'), title('Accuracy of Predictors (1 = Correct, 0 = Incorrect)')
subplot(2,2,4), bar(accuracy2,'b'), ylabel('Accuracy'), set(gca,'xticklabel',{'SVM'})
title('Total Accuracy of SVM Classification on Genders')

%% Cross-Validation of the Two Supervised Learning Algorithms

% CV of face classification
cv1 = crossval(svm1); % 10-fold cross-validation
error1 = kfoldLoss(cv1); % finding general cross-validation error

% CV of gender classification
cv2 = crossval(svm2); % 10-fold cross-validation
error2 = kfoldLoss(cv2); % finding general cross-validation error

% plotting the cross-validation error
figure(7)
bar_x = categorical({'Face Classification','Gender Classification'}); % x-axis
bar_y = [error1 error2]; % y-axis
bar(bar_x,bar_y,'b'), ylabel('10-fold CV Error')
title('Cross-Validation Error of the Supervised Learning Methods')

%% Part 2.3: Face Identification (Test 3 - Unsupervised Algorithms)

L = 2000; % number of entries in the training set
q3 = randperm(n); % generating a random matrix for indexing
x3 = vc(1:n,1:4); % taking the first 4 modes from the signature matrix

% splitting the data into training & testing sets
xtrain3 = x3(q3(1:L),:); % training set
xtest3 = x3(q3(L+1:end),:); % testing set

% getting the correct labels for the testing set
check3 = labels2(q3(L+1:end)); % labels for testing set (2 clusters)
check4 = labels1(q3(L+1:end)); % labels for testing set (38 clusters)

% finding the classification labels using Gaussian Mixture Models (GMM)
gm1 = fitgmdist(xtrain3,2,'CovarianceType','diagonal'); % GMM with k = 2
gm2 = fitgmdist(xtrain3,38,'CovarianceType','diagonal'); % GMM with k = 38
pre3 = cluster(gm1,xtest3); % classification labels from GMM (k = 2)
pre4 = cluster(gm2,xtest3); % classification labels from GMM (k = 38)

% plotting the GMM classifications
figure(8)
subplot(2,1,1), bar(pre3,'b'), hold on, plot(check3,'ro','Linewidth',[2])
xlabel('Testing Data'), ylabel('Clusters')
title('Gaussian Mixture Model with 2 Components')
legend({'GMM','Truth Labels'},'Location','EastOutside')
subplot(2,1,2), bar(pre4,'b'), hold on, plot(check4,'ro','Linewidth',[2])
xlabel('Testing Data'), ylabel('Clusters')
title('Gaussian Mixture Model with 38 Components')
legend({'GMM','Truth Labels'},'Location','EastOutside')

% determining the accuracy of the classifications (2 clusters)
count3 = 0; % pre-setting a total count for accurate predictions
for i = 1:length(pre3)
    if pre3(i) == check3(i) % if the prediction was accurate
        count3 = count3 + 1; % add 1 to the count of accurate predictions
    end
end
accuracy3 = count3/length(pre3); % dividing the count by the total number of predictions

% determining the accuracy of the classifications (38 clusters)
count4 = 0; % pre-setting a total count for accurate predictions
for i = 1:length(pre4)
    if pre4(i) == check4(i) % if the prediction was accurate
        count4 = count4 + 1; % add 1 to the count of accurate predictions
    end
end
accuracy4 = count4/length(pre4); % dividing the count by the total number of predictions

% plotting the accuracies of the GMM classifications
figure(9)
acc_x = categorical({'k = 2','k = 38'}); % x-axis
acc_y = [accuracy3 accuracy4]; % y-axis
bar(acc_x,acc_y,'b'), ylabel('Accuracy')
title('Accuracy of GMM Classifications with k Clusters')