from tqdm.auto import tqdm
from .setup_conditional_flow import *
from .eval_conditional_flow import *
from .plot_AR import *
from .get_data_AR import gaussian_smoothing_spike

def train_conditional_flow(n_epochs,
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
    
    savepath, plot_savepath, net_savepath = paths
    
    val_loss_list = []
    val_loss_flow_list = []
    best_epoch = 0
    
    pbar = tqdm(total=n_epochs)
    val_loss = 0
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
            rnn_out, hidden = encoder(data_input) # get latent representation
            conditional = torch.cat([rnn_out, time], -1)
            loss_flow = -flow_net.log_probs(target, cond_inputs=conditional).mean() # train normalizing flow
            loss_flow.backward()
            optimizer.step()
            
            pbar.set_description("Epoch-batch: {}-{} | loss: {:.2f} | val loss: {:.2f} Best Epoch: {}".format(
                i+1, 
                batch_idx+1,
                np.round(loss_flow.detach().cpu().numpy(), 2),
                np.round(val_loss, 2),
                best_epoch
            ))
        if (i+1)%1 == 0:
            flow_loss = validate_conditional_flow(encoder, flow_net, linear_transform,
                                                 val_loader, device,smooth=smooth)
            val_loss = flow_loss
            if len(val_loss_flow_list) == 1 or \
                (len(val_loss_flow_list) > 1 and np.min(val_loss_flow_list[1:]) > flow_loss):
                torch.save(encoder.state_dict(),  net_savepath + "/encoder.pt")
                torch.save(flow_net.state_dict(), net_savepath + "/flow_net.pt")
                best_epoch = i+1
                
            val_loss_list.append(flow_loss)
            val_loss_flow_list.append(flow_loss)
            
        if (i+1)%(n_epochs/5) == 0:
            with torch.no_grad():
                data_gen = []
                crps_list = []
                for stimuli, d_spike, d_smooth in zip(q, data_spike, data_smooth):
                    spike_train = generate_spike_train_conditonal_flow(encoder, flow_net,linear_transform,
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
                                                                    smooth=smooth)
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
                    crps_list.append(crps)
                    
                plot_spike_compare(data_spike, 
                                   data_gen, 
                                   important_index,
                                   plot_savepath, 
                                   i+1,
                                   q, 
                                   target=target_neuron)
                
                print("CRPS by Stimuli: {}".format(np.round(crps_list, 3)))
                
        encoder.train()
        flow_net.train()
            
        pbar.update(1) 
    pbar.close()

    return val_loss_list, best_epoch



def generate_spike_train_conditonal_flow(encoder, flow_net, linear_transform, 
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
    data_smooth = data_smooth.unsqueeze(0)
    data_spike = data_spike.unsqueeze(0)
    time_scale = 10**time_resolution
    spike_train_list = []
    for i in range(num_of_spike_train):
        spike_train = torch.clone(data_spike)
        spike_train[...,target_neuron] = torch.zeros(spike_train[...,target_neuron].shape)
        q = q.to(device)

        window_in = torch.ones(1, window_size, n_neurons, device=device)*filler
        spike_time = torch.tensor([[0.]]).to(device)
        while spike_time <=1:
            # generate interarrival time
            window_in = window_in.to(device)
            rnn_out, _ = encoder(window_in.float(), None)
            conditional = torch.cat([rnn_out, spike_time], -1)
            # sample interarrival time
            interarrival_sample = -1
            while interarrival_sample<0 or torch.isnan(interarrival_sample):
                interarrival_sample = flow_net.sample(cond_inputs=conditional)/scaling_factor
            # add to last spike time
            
            spike_time += interarrival_sample
            spike_time_index = int(spike_time*time_scale)
            if spike_time > 1 or spike_time_index >= spike_train.shape[1]:
                break
            # update spike
            spike_train[:,spike_time_index, target_neuron] = 1
            
            if smooth:
                data_input = torch.from_numpy(gaussian_smoothing_spike(spike_train.squeeze(0).detach().cpu().numpy(),
                                                      time_resolution, sigma)).unsqueeze(0)
            else:
                data_input = spike_train
                
            window_in = data_input[:,spike_time_index-window_size:spike_time_index,important_index]
            if spike_time_index < window_size:
                # update next window, dependent on current spike time
                window_in = torch.zeros((1,window_size,n_neurons))
                window_in[:,0:window_size - spike_time_index,:] = filler
                window_in[:,window_size - spike_time_index:,:] = data_input[:,0:spike_time_index,important_index]
                
        spike_train_list.append(spike_train.squeeze(0).detach().cpu().numpy())
    return spike_train_list

def validate_conditional_flow(encoder, flow_net, linear_transform, val_loader, device, smooth=True):
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
            rnn_out, hidden = encoder(window_spike) # get latent representation
            #conditional = torch.cat([rnn_out, stimuli, time], -1)
            conditional = torch.cat([rnn_out, time], -1)
            loss_flow = -flow_net.log_probs(target, conditional).mean() # train normalizing flow
            val_loss_flow += loss_flow.detach().cpu().numpy()
        val_loss_flow /= (batch_idx + 1)
    return val_loss_flow



    