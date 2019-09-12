clear;
clc;
load('letter_recognition.mat');

class_number = max(size(unique(train_labels)));
feature_number = min(size(train_data));
id = 1:feature_number;

%% part 1 - IG 26 class

tic;
tree_IG = ID3_IG( train_data, train_labels, id, -1);
acc_IG = accuracy(test_data, test_labels, tree_IG);
confuse_matrix_IG = Confusion_Matrix(test_data, test_labels, tree_IG);
toc;

%% part 1 - IG One vs All

k = class_number ;
unique_label = unique(train_labels);           %give unique labels in label array

trees_one_vs_all = cell(1, k);

for i=1:k
    
    label = One_vs_All(train_labels, unique_label(i));
    trees_one_vs_all{1, i} = ID3_IG( train_data, label, id, -1);

end

acc_IG_one_vs_all = accuracy_One_vs_All(test_data, test_labels, trees_one_vs_all);

%% part 2 - GINI

tic;
tree_GINI = ID3_GINI( train_data, train_labels, id, -1);
acc_GINI = accuracy(test_data, test_labels, tree_GINI);
confuse_matrix_GINI = Confusion_Matrix(test_data, test_labels, tree_GINI);
toc;

%% part 2 - Gini One vs All

k = class_number ;
unique_label = unique(train_labels);           %give unique labels in label array

trees_one_vs_all = cell(1, k);

for i=1:k
    
    label = One_vs_All(train_labels, unique_label(i));
    trees_one_vs_all{1, i} = ID3_GINI( train_data, label, id, -1);

end

acc_GINI_one_vs_all = accuracy_One_vs_All(test_data, test_labels, trees_one_vs_all);


%% part 3 - change 2 attributes

tree_IG_2 = ID3_IG_2( train_data, train_labels, id, -1);
acc_IG_2 = accuracy(test_data, test_labels, tree_IG_2);
confuse_matrix_IG2 = Confusion_Matrix(test_data, test_labels, tree_IG_2);

%% part 4 - IG

tic;
depth = 0;
tree_IG_Overfit = ID3_IG_Overfit( train_data, train_labels, id, -1, depth);
acc_IG_Overfit = accuracy(test_data, test_labels, tree_IG_Overfit);
toc;

%% part 4 - Gini

tic;
depth = 0;
tree_GINI_Overfit = ID3_GINI_Overfit( train_data, train_labels, id, -1, depth);
acc_GINI_Overfit = accuracy(test_data, test_labels, tree_GINI_Overfit);
toc;

%% part 5 - Random Forest with 4 features

k = 4;

[datas, labels, ids] = Shuffle_Data(train_data, train_labels, k, id);
trees_4 = cell(1, k);

for i=1:k
    
    trees_4{1, i} = ID3_IG( datas{i, 1}, labels{i, 1}, ids{i, 1}, -1);

end

acc_Random_Forest_4 = Majority_Voting(test_data, test_labels, trees_4);

%% part 5 - Random Forest with 16 features

k = 2;

[datas, labels] = Shuffle_Data_2(train_data, train_labels, k);
trees_16 = cell(1, k);

for i=1:k
    
    trees_16{1, i} = ID3_IG( datas{i, 1}, labels{i, 1}, id, -1);

end

acc_Random_Forest_16 = Majority_Voting(test_data, test_labels, trees_16);