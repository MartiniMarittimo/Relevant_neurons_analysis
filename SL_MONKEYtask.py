import numpy as np
import torch

def generate_monkey_data(num_trials, fraction_validation_trials=0.2):
    
    fixation_discrete = 4
    stimulus_duration = 25
    stimulus_end = 29
    decision_discrete = 5
    max_total_duration = fixation_discrete + stimulus_duration + decision_discrete
    
    inputs = torch.zeros((num_trials, max_total_duration, 5)) # trials; time; inputs 0:fix, 1:dx, 2:sx
    targets = torch.zeros((num_trials, max_total_duration, 3)) # 0:fix, 1:dx, 2:sx
    mask = torch.ones((num_trials, max_total_duration, 1))
        
    v1s = np.array([1, 2, 3])
    p1s = np.array([0.25, 0.5, 0.75])
    v2s = np.array([1, 2, 3])
    p2s = np.array([0.25, 0.5, 0.75])
    
    for i in range(num_trials):
        
        gt = np.zeros(3)
        
        v1 = np.random.choice(v1s)
        p1 = np.random.choice(p1s)
        v2 = np.random.choice(v2s)
        p2 = np.random.choice(p2s)
        
        gt[0] = 1
        if v1*p1 > v2*p2:
            gt[1] = 1
        elif v1*p1 < v2*p2:
            gt[2] = 1
        elif v1*p1 == v2*p2:
            pip = np.random.choice([1,2])
            gt[pip] = 1
        
        inputs[i, :int(stimulus_end), 0] = 3
        inputs[i, int(fixation_discrete):int(stimulus_end), 1] = v1
        inputs[i, int(fixation_discrete):int(stimulus_end), 2] = p1
        inputs[i, int(fixation_discrete):int(stimulus_end), 3] = v2
        inputs[i, int(fixation_discrete):int(stimulus_end), 4] = p2
        
        targets[i, :int(stimulus_end), 0] = gt[0]
        targets[i, int(stimulus_end):, 1] = gt[1]
        targets[i, int(stimulus_end):, 2] = gt[2]
        
            
        #mask[i, int(max_total_duration):, 0] = 0
        #trial_values.append(fixation_discrete)
        #trial_values.append(stimulus_duration+fixation_discrete) 
        #trial_values.append(max_total_duration)
        #if gt == 0:
        #    trial_values.append("right")
        #if gt == 1:
        #    trial_values.append("left")
    #
        #values.append(trial_values)

    split_at = num_trials - int(num_trials * fraction_validation_trials)
    
    inputs_train, inputs_val = inputs[:split_at], inputs[split_at:]
    targets_train, targets_val = targets[:split_at], targets[split_at:]
    mask_train, mask_val = mask[:split_at], mask[split_at:]
    #values_train, values_valid = values[:split_at], values[split_at:]

    return inputs_train, targets_train, inputs_val, targets_val, mask_train, mask_val 