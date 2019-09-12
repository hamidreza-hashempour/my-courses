clear 
close all
clc

load('train_lables.mat');
load('train_images.mat');

load('test_labels.mat');
load('test_images.mat');

test_data = double(test_images);
test_label = transpose(double(test_labels));

train_data = double(train_images);
train_label = transpose(double(train_lables));

clear train_images train_lables test_images test_labels ;

number_input_nodes = 784;       
number_hidden_nodes = 61;        % hidden_layer nodes
number_output_nodes = 784;        % output_node

number_epoch = 5;               % number of iteration
number_Data = 100;

alpha = 1;                        % learning rate
etta = 0.5;                       % momentum term

%% Batch - whole data

[ weights_Batch, cost_Batch ] = Train_Network_Batch_Base( train_data, number_input_nodes, number_hidden_nodes, number_output_nodes, alpha, etta, number_epoch);

%%

[ weights_Stoch, cost_Stoch ] = Train_Network_SGD_Base( train_data, number_input_nodes, number_hidden_nodes, number_output_nodes, alpha, etta, number_epoch, number_Data);

%% tied weights

[ weight_Tied, cost_Tied ] = Train_Network_Tied( train_data, number_input_nodes, number_hidden_nodes, alpha, etta, number_epoch);

%% show 10 class

Show_images_for_10_class(test_data, weights);

%% KNN with encoder

new_train_data = Sigmoid(train_data * weights{1});
new_test_data = Sigmoid(test_data(1:1000,:) * weights{1});

K = [3, 5, 7, 9];           % K for KNN
error = KNN( new_train_data, train_label, new_test_data, test_label(1:1000), K);