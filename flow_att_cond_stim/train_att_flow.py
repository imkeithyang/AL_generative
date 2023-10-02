from .setup_att_flow import *
from .eval_att_flow import *
from utils import *
from tqdm.auto import tqdm

def train_att_flow(n_epochs,
          train_loader, 
          val_loader, 
          encoder, 
          flow_net,
          linear_transform,
          optimizer,
          window_size,
          n_neurons,
          data_spike,
          data_smooth,
          q,
          time_resolution,
          filler,
          device,
          paths,
          target_neuron,
          important_index,
          scaling_factor,
          sigma,
          smooth=True):
    
    # paths to save different things
    savepath, plot_savepath, net_savepath = paths
    
    # validation metric
    val_loss_list = []
    val_crps_list = []
    val_isi_dist_list = []
    val_spike_dist_list = []
    
    best_epoch = 0
    
    pbar = tqdm(total=n_epochs)
    val_loss = 0
    val_crps = 0
    val_isi_dist_mean = 0
    val_spike_dist_mean = 0
    
    # Training Loops
    for i in range(n_epochs):
        
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            window_spike = data[0].to(device)
            window_smooth = data[1].to(device)
            stimuli = data[2].to(device)
            target = data[3].to(device)
            time = data[4].to(device)
            #rnn_out, hidden = encoder(window_spike) # get latent representation
            if smooth:
                data_input = window_smooth
            else:
                data_input = window_spike
                
            rnn_out, hidden,betai = encoder(data_input,stimuli) # get latent representation
            conditional = torch.cat([rnn_out, stimuli, time], -1) # NF condition on context, neuron, and stimuli
            loss_flow = -flow_net.log_probs(target, 
                                            cond_inputs=conditional).mean() # train normalizing flow
            loss_flow.backward()
            optimizer.step()
            
            pbar.set_description("Epoch: {} | loss: {:.2f} | val loss: {:.2f} | CRPS: {:.3f} | ISI: {:.3f} | SPIKE : {:.3f} Best Epoch: {}".format(
                i+1, 
                np.round(loss_flow.detach().cpu().numpy(), 2),
                np.round(val_loss, 2),
                np.round(val_crps, 3),
                np.round(val_isi_dist_mean, 3),
                np.round(val_spike_dist_mean, 3),
                best_epoch
            ))
            
        # validation
        if (i+1)%1 == 0:
            flow_loss = validate_att_flow(encoder, flow_net, linear_transform,
                                                 val_loader, device,smooth=smooth)
            val_loss = flow_loss
            if len(val_loss_list) == 1 or \
                (len(val_loss_list) > 1 and np.min(val_loss_list[1:]) > flow_loss):
                torch.save(encoder.state_dict(),  net_savepath + "/encoder.pt")
                torch.save(flow_net.state_dict(), net_savepath + "/flow_net.pt")
                best_epoch = i+1
                
            val_loss_list.append(flow_loss)

            
            with torch.no_grad():
                data_gen = []
                temp_crps_list = []
                temp_isi_dist_list = []
                temp_spike_dist_list = []
                # spike train genereation for all stimuli
                for stimuli, d_spike, d_smooth in zip(q, data_spike, data_smooth):
                    spike_train = generate_spike_train_att_flow(encoder, flow_net,linear_transform,
                                                                device,
                                                                target_neuron,
                                                                important_index,
                                                                window_size, 
                                                                n_neurons, 
                                                                torch.from_numpy(stimuli).float(), 
                                                                time_resolution,
                                                                sigma,
                                                                torch.from_numpy(d_spike).float(), 
                                                                torch.from_numpy(d_smooth).float(), 
                                                                scaling_factor=scaling_factor,
                                                                filler=filler,
                                                                smooth=smooth,
                                                                num_of_spike_train=1)
                    
                    # evaluating the generated spike trains using crps
                    data_gen.append(spike_train[0])
                    crps = evaluate_crps(encoder, flow_net,linear_transform,
                                  device,
                                  target_neuron,
                                  important_index,
                                  window_size,
                                  time_resolution,
                                  d_spike,
                                  d_smooth,
                                  stimuli,
                                  filler=filler,
                                  sigma=sigma,
                                  scaling_factor=scaling_factor,
                                  smooth=smooth,
                                  num_samples=2000)
                    
                    # evaluating the generated spike trains using spike distances
                    isi_dist, spike_dist = evaluate_spike_distance(d_spike, 
                                                                   spike_train, 
                                                                   target_neuron, 
                                                                   time_resolution)
                    
                    temp_crps_list.append(crps)
                    temp_isi_dist_list.append(np.mean(isi_dist))
                    temp_spike_dist_list.append(np.mean(spike_dist))
                    
                    
                val_crps = np.mean(temp_crps_list)
                val_isi_dist_mean = np.mean(temp_isi_dist_list)
                val_spike_dist_mean = np.mean(temp_spike_dist_list)
                    
                val_isi_dist_list.append(temp_isi_dist_list)
                val_spike_dist_list.append(temp_spike_dist_list)
                val_crps_list.append(temp_crps_list)
                
            # some plotting
            #if (i+1)%(n_epochs/5) == 0:
                #plot_spike_compare(data_spike, 
                #                   data_gen, 
                #                   important_index,
                #                   plot_savepath, 
                #                   i+1,
                #                   q, 
                #                   target=target_neuron)
                
        encoder.train()
        flow_net.train()
            
        pbar.update(1) 
    pbar.close()
    val_dict = {"val_loss":val_loss_list, "val_crps":val_crps_list, "val_isi":val_isi_dist_list, "val_spike":val_spike_dist_list}
    return val_dict, best_epoch



def generate_spike_train_att_flow(encoder, flow_net, linear_transform, 
                                         device, 
                                         target_neuron,
                                         important_index,
                                         window_size, 
                                         n_neurons, 
                                         q, 
                                         time_resolution, 
                                         sigma,
                                         data_spike, data_smooth, 
                                         scaling_factor = 1, filler=-1, smooth=True, num_of_spike_train = 1):
    """
    Generate spike trains given stimuli
    Returns:
        spike_train_list: a list of generated spike trains
    """
    data_smooth = data_smooth.unsqueeze(0)
    data_spike = data_spike.unsqueeze(0)
    time_scale = 10**time_resolution
    spike_train_list = []
    for i in range(num_of_spike_train):
        spike_train = torch.clone(data_spike)
        spike_train[...,target_neuron] = torch.zeros(spike_train[...,target_neuron].shape)
        q = q.to(device)

        hidden=None
        window_in = torch.ones(1, window_size, n_neurons, device=device)*filler
        spike_time = torch.tensor([[0.]]).to(device)
        while spike_time <=1.2:
            # generate interarrival time
            window_in = window_in.to(device)
            rnn_out, hidden,_ = encoder(window_in.float(), q, None)
            conditional = torch.cat([rnn_out, q, spike_time], -1)
            interarrival_sample = -1
            while interarrival_sample<0 or torch.isnan(interarrival_sample):
                interarrival_samples = flow_net.sample(cond_inputs=conditional.repeat(100,1))/scaling_factor
                interarrival_sample = interarrival_samples[interarrival_samples > 0][0]
            # add to last spike time
            
            spike_time += interarrival_sample
            spike_time_index = int(spike_time*time_scale)
            if spike_time_index >= spike_train.shape[1]:
                break
            # update spike
            spike_train[:,spike_time_index, target_neuron] = 1
            
            window_in = spike_train[:,spike_time_index-window_size:spike_time_index,important_index]
            if smooth:
                window_in = torch.from_numpy(
                    gaussian_smoothing_spike(window_in.squeeze(0).detach().cpu().numpy(),time_resolution, sigma)
                    ).unsqueeze(0)
            if spike_time_index < window_size:
                # update next window, dependent on current spike time
                window_in = torch.zeros((1,window_size,n_neurons))
                window_in[:,0:window_size - spike_time_index,:] = filler
                window_in[:,window_size - spike_time_index:,:] = spike_train[:,0:spike_time_index,important_index]
                if smooth:
                    window_in[:,window_size - spike_time_index:,:] = torch.from_numpy(
                        gaussian_smoothing_spike(spike_train[:,0:spike_time_index,important_index].squeeze(0).detach().cpu().numpy(),time_resolution, sigma)
                        ).unsqueeze(0)

        spike_train_list.append(spike_train.squeeze(0).detach().cpu().numpy())
        
        del spike_train
        torch.cuda.empty_cache()
        
    return spike_train_list

def validate_att_flow(encoder, flow_net, linear_transform, val_loader, device, smooth=True):
    encoder.eval()
    flow_net.eval()

    val_loss_flow = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            window_spike = data[0].to(device)
            window_smooth = data[1].to(device)
            stimuli = data[2].to(device)
            target = data[3].to(device)
            time = data[4].to(device)
            if smooth:
                data_input = window_smooth
            else:
                data_input = window_spike
            rnn_out, hidden,_ = encoder(data_input, stimuli)
            conditional = torch.cat([rnn_out, stimuli, time], -1)
            loss_flow = -flow_net.log_probs(target, conditional).mean()
            val_loss_flow += loss_flow.detach().cpu().numpy()
        val_loss_flow /= (batch_idx + 1)
    return val_loss_flow



    