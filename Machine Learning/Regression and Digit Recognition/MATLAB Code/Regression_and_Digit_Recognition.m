clear all; close all; clc

% loading the data set
test_images = loadMNISTImages('t10k-images.idx3-ubyte'); % testing set images
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte'); % testing set labels
train_images = loadMNISTImages('train-images.idx3-ubyte'); % training set images
train_labels = loadMNISTLabels('train-labels.idx1-ubyte'); % training set labels

%% one hot encoding the B matrix from the labels

y1 = [1 0 0 0 0 0 0 0 0 0]'; % label = 1
y2 = [0 1 0 0 0 0 0 0 0 0]'; % label = 2
y3 = [0 0 1 0 0 0 0 0 0 0]'; % label = 3
y4 = [0 0 0 1 0 0 0 0 0 0]'; % label = 4
y5 = [0 0 0 0 1 0 0 0 0 0]'; % label = 5
y6 = [0 0 0 0 0 1 0 0 0 0]'; % label = 6
y7 = [0 0 0 0 0 0 1 0 0 0]'; % label = 7
y8 = [0 0 0 0 0 0 0 1 0 0]'; % label = 8
y9 = [0 0 0 0 0 0 0 0 1 0]'; % label = 9
y10 = [0 0 0 0 0 0 0 0 0 1]'; % label = 0

B = zeros(length(train_labels),10); % pre-allocating B matrix for training set
% generating B matrix
for i = 1:length(train_labels)
    if train_labels(i) == 1
        B(i,:) = y1;
    end
    if train_labels(i) == 2
        B(i,:) = y2;
    end
    if train_labels(i) == 3
        B(i,:) = y3;
    end
    if train_labels(i) == 4
        B(i,:) = y4;
    end
    if train_labels(i) == 5
        B(i,:) = y5;
    end
    if train_labels(i) == 6
        B(i,:) = y6;
    end
    if train_labels(i) == 7
        B(i,:) = y7;
    end
    if train_labels(i) == 8
        B(i,:) = y8;
    end
    if train_labels(i) == 9
        B(i,:) = y9;
    end
    if train_labels(i) == 0
        B(i,:) = y10;
    end
end

B0 = zeros(length(test_labels),10); % pre-allocating B matrix for testing set
% generating B matrix
for i = 1:length(test_labels)
    if test_labels(i) == 1
        B0(i,:) = y1;
    end
    if test_labels(i) == 2
        B0(i,:) = y2;
    end
    if test_labels(i) == 3
        B0(i,:) = y3;
    end
    if test_labels(i) == 4
        B0(i,:) = y4;
    end
    if test_labels(i) == 5
        B0(i,:) = y5;
    end
    if test_labels(i) == 6
        B0(i,:) = y6;
    end
    if test_labels(i) == 7
        B0(i,:) = y7;
    end
    if test_labels(i) == 8
        B0(i,:) = y8;
    end
    if test_labels(i) == 9
        B0(i,:) = y9;
    end
    if test_labels(i) == 0
        B0(i,:) = y10;
    end
end

%% computing Ax = b using 3 solvers: pinv, lasso, & ridge

% psuedo-inverse method
A = pinv(train_images.'); % pseudo-inverse of images
X = A*B; % AX = B or X = pinv(A)*B

% lasso method
X0 = zeros(size(X)); % pre-allocating the X matrix
for j = 1:10
    [Xtemp,stats] = lasso(train_images.',B(:,j),'Lambda',0.1,'CV',10); % computing the lasso method of each column of B
    X0(:,j) = Xtemp; % storing the result in each column of X
end

% ridge method
X1 = zeros(size(X)); % pre-allocating the X matrix
for jj = 1:10
    Xtemp1 = ridge(B(:,jj),train_images.',0.1); % computing the ridge method of each column of B
    X1(:,j) = Xtemp1; % storing the result in each column of X
end

%% plotting absolute values of solution X

figure(1) % pseudo-inverse method
bar(abs(X),'g','EdgeColor','green'), xlabel('Pixel'), ylabel('Magnitude')
title('Absolute Value of X from Pseudo-Inverse Method')

figure(2) % lasso method
bar(abs(X0),'g','EdgeColor','green'), xlabel('Pixel'), ylabel('Magnitude')
title('Absolute Value of X from Lasso Method')

figure(3) % ridge method
bar(abs(X1),'g','EdgeColor','green'), xlabel('Pixel'), ylabel('Magnitude')
title('Absolute Value of X from Ridge Method')

%% applying important pixels to test set (pinv)

[~,I] = sort(abs(X),'descend'); % sorting pixel intensity

Z = zeros(size(X));
for i = 1:50
    for j = 1:10
        Z(I(i,j),j) = 1; % selecting the 50 most important pixels for each column
    end
end
F = test_images.'*(X.*Z); % removing the unimportant pixels from the test data

check = F == B0; % checking if the new data (A*X) is equal to the labels (B)
count = 0; % pre-setting a count for accuracy check
for j = 1:10
    for i = 1:10000
        if check(i,j) == 1
            count = count + 1; % if AX = B, add 1 to the accuracy count
        end
    end
end
accuracy = count/(10000*10); % divide the accuracy count by the total number of data points

%% applying important pixels to test set (lasso)

[~,I0] = sort(abs(X0),'descend'); % sorting pixel intensity

Z0 = zeros(size(X0));
for i = 1:50
    for j = 1:10
        Z0(I0(i,j),j) = 1; % selecting the 50 most important pixels for each column
    end
end
F0 = test_images.'*(X0.*Z0); % removing the unimportant pixels from the test data

check0 = F0 == B0; % checking if the new data (A*X) is equal to the labels (B)
count0 = 0; % pre-setting a count for accuracy check
for j = 1:10
    for i = 1:10000
        if check0(i,j) == 1
            count0 = count0 + 1; % if AX = B, add 1 to the accuracy count
        end
    end
end
accuracy0 = count0/(10000*10); % divide the accuracy count by the total number of data points

%% applying important pixels to test set (ridge)

[~,I1] = sort(abs(X1),'descend'); % sorting pixel intensity

Z1 = zeros(size(X1));
for i = 1:50
    for j = 1:10
        Z1(I1(i,j),j) = 1; % selecting the 50 most important pixels for each column
    end
end
F1 = test_images.'*(X1.*Z1); % removing the unimportant pixels from the test data

check1 = F1 == B0; % checking if the new data (A*X) is equal to the labels (B)
count1 = 0; % pre-setting a count for accuracy check
for j = 1:10
    for i = 1:10000
        if check1(i,j) == 1
            count1 = count1 + 1; % if AX = B, add 1 to the accuracy count
        end
    end
end
accuracy1 = count1/(10000*10); % divide the accuracy count by the total number of data points

%% plotting accuracies of important pixels on test set

figure(4)
acc_x = categorical({'pinv', 'lasso', 'ridge'}); % x-axis of bar graph
acc_y = [accuracy accuracy0 accuracy1]; % y-axis of bar graph
bar(acc_x, acc_y, 'c'), ylim([0 1])
xlabel('AX = B Solvers'), ylabel('Accuracy'), title('Accuracy of Important Pixels on Test Data')

%% finding most important pixels for each digit (pinv)

% sorting for largest pixel intensity per digit
[~,a1] = sort(abs(X(:,1)),'descend'); % 1st digit
[~,a2] = sort(abs(X(:,2)),'descend'); % 2nd digit
[~,a3] = sort(abs(X(:,3)),'descend'); % 3rd digit
[~,a4] = sort(abs(X(:,4)),'descend'); % 4th digit
[~,a5] = sort(abs(X(:,5)),'descend'); % 5th digit
[~,a6] = sort(abs(X(:,6)),'descend'); % 6th digit
[~,a7] = sort(abs(X(:,7)),'descend'); % 7th digit
[~,a8] = sort(abs(X(:,8)),'descend'); % 8th digit
[~,a9] = sort(abs(X(:,9)),'descend'); % 9th digit
[~,a10] = sort(abs(X(:,10)),'descend'); % 10th digit

% finding most important pixels per digit
d1 = zeros(784,1); % pre-allocating for digit 1
d2 = zeros(784,1); % pre-allocating for digit 2
d3 = zeros(784,1); % pre-allocating for digit 3
d4 = zeros(784,1); % pre-allocating for digit 4
d5 = zeros(784,1); % pre-allocating for digit 5
d6 = zeros(784,1); % pre-allocating for digit 6
d7 = zeros(784,1); % pre-allocating for digit 7
d8 = zeros(784,1); % pre-allocating for digit 8
d9 = zeros(784,1); % pre-allocating for digit 9
d10 = zeros(784,1); % pre-allocating for digit 10

for i = 1:50
    d1(a1(i)) = 1; % selecting the 50 most important pixels for digit 1
    d2(a2(i)) = 1; % selecting the 50 most important pixels for digit 2
    d3(a3(i)) = 1; % selecting the 50 most important pixels for digit 3
    d4(a4(i)) = 1; % selecting the 50 most important pixels for digit 4
    d5(a5(i)) = 1; % selecting the 50 most important pixels for digit 5
    d6(a6(i)) = 1; % selecting the 50 most important pixels for digit 6
    d7(a7(i)) = 1; % selecting the 50 most important pixels for digit 7
    d8(a8(i)) = 1; % selecting the 50 most important pixels for digit 8
    d9(a9(i)) = 1; % selecting the 50 most important pixels for digit 9
    d10(a10(i)) = 1; % selecting the 50 most important pixels for digit 10
end

% removing the unimportant pixels from the test data
m1 = test_images.'*(X(:,1).*d1); % 1st digit
m2 = test_images.'*(X(:,2).*d2); % 2nd digit
m3 = test_images.'*(X(:,3).*d3); % 3rd digit
m4 = test_images.'*(X(:,4).*d4); % 4th digit
m5 = test_images.'*(X(:,5).*d5); % 5th digit
m6 = test_images.'*(X(:,6).*d6); % 6th digit
m7 = test_images.'*(X(:,7).*d7); % 7th digit
m8 = test_images.'*(X(:,8).*d8); % 8th digit
m9 = test_images.'*(X(:,9).*d9); % 9th digit
m10 = test_images.'*(X(:,10).*d10); % 10th digit

% checking if the new data (A*X) is equal to the labels (B)
g1 = m1 == B0(:,1); % 1st digit
g2 = m2 == B0(:,2); % 2nd digit
g3 = m3 == B0(:,3); % 3rd digit
g4 = m4 == B0(:,4); % 4th digit
g5 = m5 == B0(:,5); % 5th digit
g6 = m6 == B0(:,6); % 6th digit
g7 = m7 == B0(:,7); % 7th digit
g8 = m8 == B0(:,8); % 8th digit
g9 = m9 == B0(:,9); % 9th digit
g10 = m10 == B0(:,10); % 10th digit

% pre-setting a count for accuracy check
count1 = 0; % 1st digit
count2 = 0; % 2nd digit
count3 = 0; % 3rd digit
count4 = 0; % 4th digit
count5 = 0; % 5th digit
count6 = 0; % 6th digit
count7 = 0; % 7th digit
count8 = 0; % 8th digit
count9 = 0; % 9th digit
count10 = 0; % 10th digit

% calculating accuracy per digit
for i = 1:10000
    if g1(i) == 1 % 1st digit
        count1 = count1 + 1; % if AX = B, add 1 to the accuracy count
    end
    if g2(i) == 1 % 2nd digit
        count2 = count2 + 1; % if AX = B, add 1 to the accuracy count
    end
    if g3(i) == 1 % 3rd digit
        count3 = count3 + 1; % if AX = B, add 1 to the accuracy count
    end
    if g4(i) == 1 % 4th digit
        count4 = count4 + 1; % if AX = B, add 1 to the accuracy count
    end
    if g5(i) == 1 % 5th digit
        count5 = count5 + 1; % if AX = B, add 1 to the accuracy count
    end
    if g6(i) == 1 % 6th digit
        count6 = count6 + 1; % if AX = B, add 1 to the accuracy count
    end
    if g7(i) == 1 % 7th digit
        count7 = count7 + 1; % if AX = B, add 1 to the accuracy count
    end
    if g8(i) == 1 % 8th digit
        count8 = count8 + 1; % if AX = B, add 1 to the accuracy count
    end
    if g9(i) == 1 % 9th digit
        count9 = count9 + 1; % if AX = B, add 1 to the accuracy count
    end
    if g10(i) == 1 % 10th digit
        count10 = count10 + 1; % if AX = B, add 1 to the accuracy count
    end
end

% divide the accuracy count by the total number of data points
acc1 = count1/10000; % 1st digit
acc2 = count2/10000; % 2nd digit
acc3 = count3/10000; % 3rd digit
acc4 = count4/10000; % 4th digit
acc5 = count5/10000; % 5th digit
acc6 = count6/10000; % 6th digit
acc7 = count7/10000; % 7th digit
acc8 = count8/10000; % 8th digit
acc9 = count9/10000; % 9th digit
acc10 = count10/10000; % 10th digit

% plotting accuracies of each digit
figure(5)
acc_x = categorical({'1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'}); % x-axis of bar graph
acc_y = [acc1 acc2 acc3 acc4 acc5 acc6 acc7 acc8 acc9 acc10]; % y-axis of bar graph
bar(acc_x, acc_y, 'c'), ylim([0 1])
xlabel('Digits'), ylabel('Accuracy'), title('Accuracy of Important Pixels on Individual Digits (pinv)')

%% finding most important pixels for each digit (lasso)

% sorting for largest pixel intensity per digit
[~,b1] = sort(abs(X0(:,1)),'descend'); % 1st digit
[~,b2] = sort(abs(X0(:,2)),'descend'); % 2nd digit
[~,b3] = sort(abs(X0(:,3)),'descend'); % 3rd digit
[~,b4] = sort(abs(X0(:,4)),'descend'); % 4th digit
[~,b5] = sort(abs(X0(:,5)),'descend'); % 5th digit
[~,b6] = sort(abs(X0(:,6)),'descend'); % 6th digit
[~,b7] = sort(abs(X0(:,7)),'descend'); % 7th digit
[~,b8] = sort(abs(X0(:,8)),'descend'); % 8th digit
[~,b9] = sort(abs(X0(:,9)),'descend'); % 9th digit
[~,b10] = sort(abs(X0(:,10)),'descend'); % 10th digit

% finding most important pixels per digit
e1 = zeros(784,1); % pre-allocating for digit 1
e2 = zeros(784,1); % pre-allocating for digit 2
e3 = zeros(784,1); % pre-allocating for digit 3
e4 = zeros(784,1); % pre-allocating for digit 4
e5 = zeros(784,1); % pre-allocating for digit 5
e6 = zeros(784,1); % pre-allocating for digit 6
e7 = zeros(784,1); % pre-allocating for digit 7
e8 = zeros(784,1); % pre-allocating for digit 8
e9 = zeros(784,1); % pre-allocating for digit 9
e10 = zeros(784,1); % pre-allocating for digit 10

for i = 1:50
    e1(b1(i)) = 1; % selecting the 50 most important pixels for digit 1
    e2(b2(i)) = 1; % selecting the 50 most important pixels for digit 2
    e3(b3(i)) = 1; % selecting the 50 most important pixels for digit 3
    e4(b4(i)) = 1; % selecting the 50 most important pixels for digit 4
    e5(b5(i)) = 1; % selecting the 50 most important pixels for digit 5
    e6(b6(i)) = 1; % selecting the 50 most important pixels for digit 6
    e7(b7(i)) = 1; % selecting the 50 most important pixels for digit 7
    e8(b8(i)) = 1; % selecting the 50 most important pixels for digit 8
    e9(b9(i)) = 1; % selecting the 50 most important pixels for digit 9
    e10(b10(i)) = 1; % selecting the 50 most important pixels for digit 10
end

% removing the unimportant pixels from the test data
n1 = test_images.'*(X0(:,1).*e1); % 1st digit
n2 = test_images.'*(X0(:,2).*e2); % 2nd digit
n3 = test_images.'*(X0(:,3).*e3); % 3rd digit
n4 = test_images.'*(X0(:,4).*e4); % 4th digit
n5 = test_images.'*(X0(:,5).*e5); % 5th digit
n6 = test_images.'*(X0(:,6).*e6); % 6th digit
n7 = test_images.'*(X0(:,7).*e7); % 7th digit
n8 = test_images.'*(X0(:,8).*e8); % 8th digit
n9 = test_images.'*(X0(:,9).*e9); % 9th digit
n10 = test_images.'*(X0(:,10).*e10); % 10th digit

% checking if the new data (A*X) is equal to the labels (B)
h1 = n1 == B0(:,1); % 1st digit
h2 = n2 == B0(:,2); % 2nd digit
h3 = n3 == B0(:,3); % 3rd digit
h4 = n4 == B0(:,4); % 4th digit
h5 = n5 == B0(:,5); % 5th digit
h6 = n6 == B0(:,6); % 6th digit
h7 = n7 == B0(:,7); % 7th digit
h8 = n8 == B0(:,8); % 8th digit
h9 = n9 == B0(:,9); % 9th digit
h10 = n10 == B0(:,10); % 10th digit

% pre-setting a count for accuracy check
count1 = 0; % 1st digit
count2 = 0; % 2nd digit
count3 = 0; % 3rd digit
count4 = 0; % 4th digit
count5 = 0; % 5th digit
count6 = 0; % 6th digit
count7 = 0; % 7th digit
count8 = 0; % 8th digit
count9 = 0; % 9th digit
count10 = 0; % 10th digit

% calculating accuracy per digit
for i = 1:10000
    if h1(i) == 1 % 1st digit
        count1 = count1 + 1; % if AX = B, add 1 to the accuracy count
    end
    if h2(i) == 1 % 2nd digit
        count2 = count2 + 1; % if AX = B, add 1 to the accuracy count
    end
    if h3(i) == 1 % 3rd digit
        count3 = count3 + 1; % if AX = B, add 1 to the accuracy count
    end
    if h4(i) == 1 % 4th digit
        count4 = count4 + 1; % if AX = B, add 1 to the accuracy count
    end
    if h5(i) == 1 % 5th digit
        count5 = count5 + 1; % if AX = B, add 1 to the accuracy count
    end
    if h6(i) == 1 % 6th digit
        count6 = count6 + 1; % if AX = B, add 1 to the accuracy count
    end
    if h7(i) == 1 % 7th digit
        count7 = count7 + 1; % if AX = B, add 1 to the accuracy count
    end
    if h8(i) == 1 % 8th digit
        count8 = count8 + 1; % if AX = B, add 1 to the accuracy count
    end
    if h9(i) == 1 % 9th digit
        count9 = count9 + 1; % if AX = B, add 1 to the accuracy count
    end
    if h10(i) == 1 % 10th digit
        count10 = count10 + 1; % if AX = B, add 1 to the accuracy count
    end
end

% divide the accuracy count by the total number of data points
acc1 = count1/10000; % 1st digit
acc2 = count2/10000; % 2nd digit
acc3 = count3/10000; % 3rd digit
acc4 = count4/10000; % 4th digit
acc5 = count5/10000; % 5th digit
acc6 = count6/10000; % 6th digit
acc7 = count7/10000; % 7th digit
acc8 = count8/10000; % 8th digit
acc9 = count9/10000; % 9th digit
acc10 = count10/10000; % 10th digit

% plotting accuracies of each digit
figure(6)
acc_x = categorical({'1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'}); % x-axis of bar graph
acc_y = [acc1 acc2 acc3 acc4 acc5 acc6 acc7 acc8 acc9 acc10]; % y-axis of bar graph
bar(acc_x, acc_y, 'c'), ylim([0 1])
xlabel('Digits'), ylabel('Accuracy'), title('Accuracy of Important Pixels on Individual Digits (lasso)')

%% finding most important pixels for each digit (ridge)

% sorting for largest pixel intensity per digit
[~,c1] = sort(abs(X1(:,1)),'descend'); % 1st digit
[~,c2] = sort(abs(X1(:,2)),'descend'); % 2nd digit
[~,c3] = sort(abs(X1(:,3)),'descend'); % 3rd digit
[~,c4] = sort(abs(X1(:,4)),'descend'); % 4th digit
[~,c5] = sort(abs(X1(:,5)),'descend'); % 5th digit
[~,c6] = sort(abs(X1(:,6)),'descend'); % 6th digit
[~,c7] = sort(abs(X1(:,7)),'descend'); % 7th digit
[~,c8] = sort(abs(X1(:,8)),'descend'); % 8th digit
[~,c9] = sort(abs(X1(:,9)),'descend'); % 9th digit
[~,c10] = sort(abs(X1(:,10)),'descend'); % 10th digit

% finding most important pixels per digit
f1 = zeros(784,1); % pre-allocating for digit 1
f2 = zeros(784,1); % pre-allocating for digit 2
f3 = zeros(784,1); % pre-allocating for digit 3
f4 = zeros(784,1); % pre-allocating for digit 4
f5 = zeros(784,1); % pre-allocating for digit 5
f6 = zeros(784,1); % pre-allocating for digit 6
f7 = zeros(784,1); % pre-allocating for digit 7
f8 = zeros(784,1); % pre-allocating for digit 8
f9 = zeros(784,1); % pre-allocating for digit 9
f10 = zeros(784,1); % pre-allocating for digit 10

for i = 1:50
    f1(c1(i)) = 1; % selecting the 50 most important pixels for digit 1
    f2(c2(i)) = 1; % selecting the 50 most important pixels for digit 2
    f3(c3(i)) = 1; % selecting the 50 most important pixels for digit 3
    f4(c4(i)) = 1; % selecting the 50 most important pixels for digit 4
    f5(c5(i)) = 1; % selecting the 50 most important pixels for digit 5
    f6(c6(i)) = 1; % selecting the 50 most important pixels for digit 6
    f7(c7(i)) = 1; % selecting the 50 most important pixels for digit 7
    f8(c8(i)) = 1; % selecting the 50 most important pixels for digit 8
    f9(c9(i)) = 1; % selecting the 50 most important pixels for digit 9
    f10(c10(i)) = 1; % selecting the 50 most important pixels for digit 10
end

% removing the unimportant pixels from the test data
p1 = test_images.'*(X1(:,1).*f1); % 1st digit
p2 = test_images.'*(X1(:,2).*f2); % 2nd digit
p3 = test_images.'*(X1(:,3).*f3); % 3rd digit
p4 = test_images.'*(X1(:,4).*f4); % 4th digit
p5 = test_images.'*(X1(:,5).*f5); % 5th digit
p6 = test_images.'*(X1(:,6).*f6); % 6th digit
p7 = test_images.'*(X1(:,7).*f7); % 7th digit
p8 = test_images.'*(X1(:,8).*f8); % 8th digit
p9 = test_images.'*(X1(:,9).*f9); % 9th digit
p10 = test_images.'*(X1(:,10).*f10); % 10th digit

% checking if the new data (A*X) is equal to the labels (B)
r1 = p1 == B0(:,1); % 1st digit
r2 = p2 == B0(:,2); % 2nd digit
r3 = p3 == B0(:,3); % 3rd digit
r4 = p4 == B0(:,4); % 4th digit
r5 = p5 == B0(:,5); % 5th digit
r6 = p6 == B0(:,6); % 6th digit
r7 = p7 == B0(:,7); % 7th digit
r8 = p8 == B0(:,8); % 8th digit
r9 = p9 == B0(:,9); % 9th digit
r10 = p10 == B0(:,10); % 10th digit

% pre-setting a count for accuracy check
count1 = 0; % 1st digit
count2 = 0; % 2nd digit
count3 = 0; % 3rd digit
count4 = 0; % 4th digit
count5 = 0; % 5th digit
count6 = 0; % 6th digit
count7 = 0; % 7th digit
count8 = 0; % 8th digit
count9 = 0; % 9th digit
count10 = 0; % 10th digit

% calculating accuracy per digit
for i = 1:10000
    if r1(i) == 1 % 1st digit
        count1 = count1 + 1; % if AX = B, add 1 to the accuracy count
    end
    if r2(i) == 1 % 2nd digit
        count2 = count2 + 1; % if AX = B, add 1 to the accuracy count
    end
    if r3(i) == 1 % 3rd digit
        count3 = count3 + 1; % if AX = B, add 1 to the accuracy count
    end
    if r4(i) == 1 % 4th digit
        count4 = count4 + 1; % if AX = B, add 1 to the accuracy count
    end
    if r5(i) == 1 % 5th digit
        count5 = count5 + 1; % if AX = B, add 1 to the accuracy count
    end
    if r6(i) == 1 % 6th digit
        count6 = count6 + 1; % if AX = B, add 1 to the accuracy count
    end
    if r7(i) == 1 % 7th digit
        count7 = count7 + 1; % if AX = B, add 1 to the accuracy count
    end
    if r8(i) == 1 % 8th digit
        count8 = count8 + 1; % if AX = B, add 1 to the accuracy count
    end
    if r9(i) == 1 % 9th digit
        count9 = count9 + 1; % if AX = B, add 1 to the accuracy count
    end
    if r10(i) == 1 % 10th digit
        count10 = count10 + 1; % if AX = B, add 1 to the accuracy count
    end
end

% divide the accuracy count by the total number of data points
acc1 = count1/10000; % 1st digit
acc2 = count2/10000; % 2nd digit
acc3 = count3/10000; % 3rd digit
acc4 = count4/10000; % 4th digit
acc5 = count5/10000; % 5th digit
acc6 = count6/10000; % 6th digit
acc7 = count7/10000; % 7th digit
acc8 = count8/10000; % 8th digit
acc9 = count9/10000; % 9th digit
acc10 = count10/10000; % 10th digit

% plotting accuracies of each digit
figure(7)
acc_x = categorical({'1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'}); % x-axis of bar graph
acc_y = [acc1 acc2 acc3 acc4 acc5 acc6 acc7 acc8 acc9 acc10]; % y-axis of bar graph
bar(acc_x, acc_y, 'c'), ylim([0 1])
xlabel('Digits'), ylabel('Accuracy'), title('Accuracy of Important Pixels on Individual Digits (ridge)')