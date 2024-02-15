% Burst detection using Poisson surprise as the detection criteria.
% Author: Prasad Gabbur
% Documentor: Prasad Gabbur
%
% The input parameter 'time_stamps' is a vector of time instants of spike
% events. It is assumed that the first value in the vector is
% 0 (reference). However it is not considered to be a spike event in
% the calculations. The parameter 'min_surprise' is the minimum value
% of Poisson surprise for a burst to be considered significant. Note 
% that the surprise threshold depends on the base of the log used in the 
% equation for computing surprise. The base used here is 10. The user 
% should note this when deciding the threshold value. The parameter
% 'max_num_burst_spikes' can be used to specify the maximum number of spikes
% that a burst can be expected to have. It is helpful to avoid precision
% problems when the probability of observing more than a certain number of
% spikes in a burst tends to zero. This parameter can be made ineffective by
% setting it to 'inf'.
% The output parameter 'burst_indicator' has 1's at the positions of spikes that
% belong to one of the detected bursts. At other spike positions it has a value
% of 0. The size of the 'burst_indicator' array is one less than the size of
% the 'time_stamps' array because of the 0 reference not being considered as a
% spike.
% The output parameter 'final_num_bursts' denotes the detected number of
% bursts.
% The output parameter 'final_burst_ranges' holds the start and end indices
% of each burst in the spike train. Each row corresponds to one burst.
% In other words, each row denotes an index range in the 'burst_indicator'
% vector within which all the elements are 1. Note that the index ranges
% might be continuous, i.e., spike trains belonging to two different bursts
% might be adjacent to each other.
% The output parameter 'final_burst_surprises' contains the surprise values
% for the detected bursts. It is of the same size as the number of detected
% bursts.

function [burst_indicator, final_num_bursts, final_burst_ranges, final_burst_surprises,total_time] = detect_bursts(time_stamps, min_surprise,max_num_burst_spikes)

num_spikes      = size(time_stamps, 1) - 1;
burst_indicator = zeros(num_spikes, 1);
total_time      = time_stamps(num_spikes + 1);

intervals = diff(time_stamps);
mean_freq = num_spikes/total_time;
mean_isi  = 1/mean_freq;

burst_ranges    = [];
burst_surprises = [];

crop_burst_ranges    = [];
crop_burst_surprises = [];

final_burst_ranges    = [];
final_burst_surprises = [];

% Find potential bursts:
% 1. Check for two consecutive ISI < 0.5*mean_ISI
% 2. Include consecutive spikes until ISI > mean_ISI
% 3. Compute surprise for each new inclusion
% 4. Retain the spike train that has the maximum surprise
i = 1;
while (i < num_spikes)
    burst_end_index = i;
    
    if ( (intervals(i) < (0.5*mean_isi)) & (intervals(i+1) < (0.5*mean_isi)) )
        if (i == 1)
            burst_start_index = 1;
        else
            burst_start_index = i - 1;
        end
        
        max_surprise = 0;
        
        period = intervals(i) + intervals(i+1);
        number = 2;
        surprise = poisson_surprise(number, mean_freq, period);
        
        if (surprise >= max_surprise)
            burst_end_index = i+1;
            max_surprise    = surprise;
        end
        
        j = i + 2;
        
        while ( (j <= num_spikes) & (intervals(j) <= mean_isi) & (number <= max_num_burst_spikes) )
            period   = period + intervals(j);
            number   = number + 1;
            surprise = poisson_surprise(number, mean_freq, period);
            
            if (surprise >= max_surprise)
                burst_end_index = j;
                max_surprise    = surprise;
            end
            
            j = j+1;
        end
        
        burst_ranges    = [burst_ranges; burst_start_index burst_end_index];
        burst_surprises = [burst_surprises; max_surprise];
    end
    
    i = burst_end_index + 1;
end

% Debugging
burst_ranges;
burst_surprises;

num_bursts = size(burst_ranges, 1);

% Maximize surprise within the detected bursts by cropping spikes at the beginning.
for i = 1:num_bursts
    num_burst_spikes = burst_ranges(i, 2) - burst_ranges(i, 1) + 1;

    surprise         = burst_surprises(i);
    max_surprise     = surprise;
    crop_start_index = burst_ranges(i, 1);

    start_index      = burst_ranges(i, 1) + 1;
    end_index        = burst_ranges(i, 2);

    while (start_index < end_index)
        period   = sum(intervals((start_index+1):end_index));
        number   = end_index - start_index;
        surprise = poisson_surprise(number, mean_freq, period);

        if (surprise >= max_surprise)
            crop_start_index = start_index;
            max_surprise     = surprise;
        end

        start_index = start_index + 1;
    end

    crop_burst_ranges    = [crop_burst_ranges; crop_start_index end_index];
    crop_burst_surprises = [crop_burst_surprises; max_surprise];
end

% Debugging
crop_burst_ranges;
crop_burst_surprises;

crop_num_bursts = size(crop_burst_ranges, 1);

% Retain bursts with at least 3 spikes in them.
for i = 1:crop_num_bursts
    num_burst_spikes = crop_burst_ranges(i, 2) - crop_burst_ranges(i, 1) + 1;
    surprise         = crop_burst_surprises(i);

    if (num_burst_spikes >= 3)
        if (surprise >= min_surprise)
            final_burst_ranges    = [final_burst_ranges; crop_burst_ranges(i, :)];
            final_burst_surprises = [final_burst_surprises; crop_burst_surprises(i)];
            
            burst_indicator(crop_burst_ranges(i, 1):crop_burst_ranges(i, 2)) = 1;
        end
    end
end

% Debugging
final_burst_ranges;
final_burst_surprises;
burst_indicator;

final_num_bursts = size(final_burst_ranges, 1);
