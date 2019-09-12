clc
clear
number_iteration = 1000;
epsilon =0.1;
alpha = 1;
gamma = 0.8;

%% 2

R_2 = R_matrix(2);
[Q_2, Q2_avg] = Q_matrix(R_2, alpha, gamma, epsilon, [0 0], number_iteration, [2 2]);
P_2 = Policy(Q_2, [0 0], [2 2]);

%% 3

R_3 = R_matrix(3);
[Q_3, Q3_avg] = Q_matrix(R_3, alpha, gamma, epsilon, [0 0 0], number_iteration, [2 2 2]);
P_3 = Policy(Q_3, [0 0 0], [2 2 2]);

%% 4

R_4 = R_matrix(4);
[Q_4, Q4_avg] = Q_matrix(R_4, alpha, gamma, epsilon, [0 0 0 0], number_iteration, [2 2 2 2]);
P_4 = Policy(Q_4, [0 0 0 0], [2 2 2 2]);

%% 5

R_5 = R_matrix(5);
[Q_5, Q5_avg] = Q_matrix(R_5, alpha, gamma, epsilon, [0 0 0 0 0], number_iteration, [2 2 2 2 2]);
P_5 = Policy(Q_5, [0 0 0 0 0], [2 2 2 2 2]);

%% change parameters

alpha = 0.1;
[Q_31, Q31_avg] = Q_matrix(R_3, alpha, gamma, epsilon, [0 0 0], number_iteration, [2 2 2]);

alpha = 0.5;
[Q_32, Q32_avg] = Q_matrix(R_3, alpha, gamma, epsilon, [0 0 0], number_iteration, [2 2 2]);

%% 
alpha = 1;

gamma = 0.5;
[Q_33, Q33_avg] = Q_matrix(R_3, alpha, gamma, epsilon, [0 0 0], number_iteration, [2 2 2]);

gamma = 0.1;
[Q_34, Q34_avg] = Q_matrix(R_3, alpha, gamma, epsilon, [0 0 0], number_iteration, [2 2 2]);
