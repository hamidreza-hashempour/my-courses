train_data = load('data_train');
test_data = load('data_test');
train_data = train_data.data_train;
test_data = test_data.data_test;
positive_train_instances = (train_data(train_data(:,6)==1,1:5));
negative_train_instances = (train_data(train_data(:,6)==0,1:5));
data_size = length(train_data);
positive_train_instances_size = length(positive_train_instances);
prior_probability = positive_train_instances_size/data_size %prior probability - part "a" answer

ill_prob = prior_probability;
not_ill_prob = 1 - prior_probability;
min(positive_train_instances(:,1))
p_x1_condition_1 = fitdist(positive_train_instances(:,1),'Normal');
p_x2_condition_1 = fitdist(positive_train_instances(:,2),'Normal');
p_x3_condition_1 = fitdist(positive_train_instances(:,3),'Normal');
p_x4_condition_1 = fitdist(positive_train_instances(:,4),'Normal');
p_x5_condition_1 = fitdist(positive_train_instances(:,5),'Normal');

p_x1_condition_0 = fitdist(negative_train_instances(:,1),'Normal');
p_x2_condition_0 = fitdist(negative_train_instances(:,2),'Normal');
p_x3_condition_0 = fitdist(negative_train_instances(:,3),'Normal');
p_x4_condition_0 = fitdist(negative_train_instances(:,4),'Normal');
p_x5_condition_0 = fitdist(negative_train_instances(:,5),'Normal');

correct_answers = 0;
for test_data_id = 1:length(test_data)
    %calc_positive_probability 
    positive_prob = ill_prob*p_x1_condition_1.pdf(test_data(test_data_id,1))*p_x2_condition_1.pdf(test_data(test_data_id,2))*p_x3_condition_1.pdf(test_data(test_data_id,3))*p_x4_condition_1.pdf(test_data(test_data_id,4));
    %calc_negative_probability 
    negative_prob = not_ill_prob.*p_x1_condition_0.pdf(test_data(test_data_id,1)).*p_x2_condition_0.pdf(test_data(test_data_id,2)).*p_x3_condition_0.pdf(test_data(test_data_id,3)).*p_x4_condition_0.pdf(test_data(test_data_id,4));
    label = 0;
    if(positive_prob>negative_prob)
        label = 1;
    end
    if(label == test_data(test_data_id,6))
        correct_answers=correct_answers+1;
    end
end

train_correct_answer=0;
positive_detected_count=0;
positive_detected_and_positive=0;
for train_data_id =1:length(train_data)
    %calc_positive_probability 
    positive_prob = ill_prob*p_x1_condition_1.pdf(train_data(train_data_id,1))*p_x2_condition_1.pdf(train_data(train_data_id,2))*p_x3_condition_1.pdf(train_data(train_data_id,3))*p_x4_condition_1.pdf(train_data(train_data_id,4));
    %calc_negative_probability 
    negative_prob = not_ill_prob*p_x1_condition_0.pdf(train_data(train_data_id,1))*p_x2_condition_0.pdf(train_data(train_data_id,2))*p_x3_condition_0.pdf(train_data(train_data_id,3))*p_x4_condition_0.pdf(train_data(train_data_id,4));
    label = 0;
    if(positive_prob>negative_prob)
        positive_detected_count=positive_detected_count+1;
        label = 1;
        if(label == train_data(train_data_id,6))
            positive_detected_and_positive=positive_detected_and_positive+1;
        end
    end
    if(label == train_data(train_data_id,6))
        train_correct_answer=train_correct_answer+1;
    end
end

correct_answers
train_accuracy=train_correct_answer/10000
accuracy = correct_answers/1000
positive_detect_prob = positive_detected_count/10000
positive_detected_and_positive_prob = positive_detected_and_positive/positive_train_instances_size
part3_ans = (positive_detected_and_positive_prob*prior_probability)/positive_detect_prob



