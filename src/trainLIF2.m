%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nf = 15;
fs = 16500;
fel = 60;
Nt_max = 10000;
th = 100;
ID = 5;

T_sim = 50;
max_rate = 1;

tau = 10;
V_th = 0.5;
lr = 1e-3;
epochs = 5;

A_plus  = 0.01;
A_minus = 0.012;

lambda = 0.01;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Feature Toggles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
USE_TEMPORAL_TOLERANCE = true;
USE_LABEL_SMOOTHING    = true;
USE_RATE_OUTPUT        = true;
USE_ELIGIBILITY_TRACE  = true;
USE_LONGER_STDP        = true;
USE_INPUT_SHIFT        = false;

tol_window = 2;
smooth_kernel = [0.2 0.6 1 0.6 0.2];
elig_decay = 0.9;
shift_steps = 1;

if USE_LONGER_STDP
    tau_pre  = 20;
    tau_post = 20;
else
    tau_pre  = 10;
    tau_post = 10;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load("C:\Users\schir\OneDrive\Studium\34_Github\SpikeNILM\data\redd3HF.mat")

time = squeeze(input(1:Nt_max,1,1));
X = squeeze(input(1:Nt_max,3:end,:));
y = squeeze(output(1:Nt_max,3:end));

[Nt, M] = size(y);

if ID ~= 0
    y = y(:,ID);
    M = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Preprocess
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_ac = squeeze(X(:,:,2));
V_ac = squeeze(X(:,:,1));

Vrms = rms(V_ac')';
Irms = rms(I_ac')';
Pact = Vrms .* Irms;

I_f_ac = abs(fft(I_ac'))' / (fs/fel);
I_f_ac = 2 * I_f_ac(:,2:Nf+1);

dI_f_ac = [zeros(1,Nf); abs(diff(I_f_ac))];

if USE_INPUT_SHIFT
    dI_f_ac = [zeros(shift_steps,Nf); dI_f_ac(1:end-shift_steps,:)];
end

dydt = [zeros(1,M); diff(y)];
s = abs(dydt);
s(s < th) = 0;
s(s >= th) = 1;

% Label smoothing
if USE_LABEL_SMOOTHING
    s_smooth = zeros(size(s));
    for m = 1:M
        s_smooth(:,m) = conv(s(:,m), smooth_kernel, 'same');
    end
    s_smooth = min(s_smooth,1);
else
    s_smooth = s;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Encoding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dI_norm = dI_f_ac ./ ((th./Vrms) + 1e-12);

spikes = zeros(T_sim, Nf, Nt);

for t = 1:Nt
    for f = 1:Nf
        rate = dI_norm(t,f) * max_rate;
        spikes(:,f,t) = rand(T_sim,1) < rate;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Physics Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P_nodes = mean(y,1);
P_nodes = P_nodes / (max(P_nodes)+1e-12);

W_phys = zeros(Nf,M);

for m = 1:M
    for f = 1:Nf
        W_phys(f,m) = P_nodes(m) * (0.5 + f/Nf);
    end
end

W_phys = W_phys ./ (norm(W_phys)+1e-12);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = W_phys + 0.01*randn(Nf,M);

pre_trace  = zeros(Nf,1);
post_trace = zeros(M,1);
eligibility = zeros(Nf,M);

train_error = zeros(epochs,1);
train_acc   = zeros(epochs,1);

W_prev = W;
weight_change = zeros(epochs,1);
weight_norm   = zeros(epochs,1);
weight_dist   = zeros(epochs,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper: Error Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
compute_error = @(yp, s) ...
    mean(abs(apply_tolerance(yp, s, USE_TEMPORAL_TOLERANCE, tol_window) - s),'all');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STDP Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ep = 1:epochs
    
    fprintf("Epoch %d\n", ep);
    V = zeros(M,1);
    y_pred_train = zeros(Nt,M);
    
    pre_trace(:)=0; post_trace(:)=0; eligibility(:)=0;
    
    for t = 1:Nt
        
        spike_sum = zeros(M,1);
        
        for ts = 1:T_sim
            
            input_vec = squeeze(spikes(ts,:,t))';
            
            pre_trace = pre_trace * exp(-1/tau_pre) + input_vec;
            
            for m = 1:M
                
                I_in = W(:,m)' * input_vec;
                V(m) = V(m) + (-V(m) + I_in)/tau;
                
                post_spike = 0;
                if V(m) >= V_th
                    post_spike = 1;
                    spike_sum(m)=spike_sum(m)+1;
                    V(m)=V(m)-V_th;
                end
                
                post_trace(m)=post_trace(m)*exp(-1/tau_post)+post_spike;
                
                dW = A_plus*pre_trace*post_spike - A_minus*input_vec*post_trace(m);
                
                if USE_ELIGIBILITY_TRACE
                    eligibility(:,m)=elig_decay*eligibility(:,m)+pre_trace*post_spike;
                    dW = eligibility(:,m);
                end
                
                target = s_smooth(t,m);
                
                if target>0
                    W(:,m)=W(:,m)+lr*dW;
                else
                    W(:,m)=W(:,m)-lr*0.5*abs(dW);
                end
                
                W(:,m)=W(:,m)-lambda*(W(:,m)-W_phys(:,m));
            end
        end
        
        if USE_RATE_OUTPUT
            y_pred_train(t,:) = spike_sum / T_sim;
        else
            y_pred_train(t,:) = spike_sum > 0;
        end
    end
    
    train_error(ep)=compute_error(y_pred_train,s);
    train_acc(ep)=mean((y_pred_train>0)==s,'all');
    
    fprintf("Train Error: %.4f | Acc: %.4f\n",train_error(ep),train_acc(ep));
    
    W = W./(norm(W,'fro')+1e-12);
    
    dW = W-W_prev;
    weight_change(ep)=norm(dW,'fro');
    weight_norm(ep)=norm(W,'fro');
    weight_dist(ep)=norm(W-W_phys,'fro');
    W_prev=W;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inference
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V=zeros(M,1);
lif_spk=zeros(Nt*T_sim,M);

idx=1;
for t=1:Nt
    for ts=1:T_sim
        input_vec=squeeze(spikes(ts,:,t))';
        for m=1:M
            V(m)=V(m)+(-V(m)+W(:,m)'*input_vec)/tau;
            if V(m)>=V_th
                lif_spk(idx,m)=1;
                V(m)=V(m)-V_th;
            end
        end
        idx=idx+1;
    end
end

lif_cycle = reshape(lif_spk,T_sim,Nt,M);
lif_cycle = squeeze(sum(lif_cycle,1))>0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
subplot(3,1,1); plot(train_error); title("Train Error");
subplot(3,1,2); plot(weight_change); title("Weight Change");
subplot(3,1,3); plot(weight_dist); title("Distance to Physics");

figure;

subplot(4,1,1);
plot(time, Pact);
hold on
plot(time, sum(y, 2))
grid on;
title('Aggregated Power');

subplot(4,1,2);
imagesc(log(dI_f_ac)');
title('Harmonic Changes');

subplot(4,1,3);
plot(lif_time, lif_mem);
hold on;
plot(lif_time, V_th * ones(size(lif_time)));
grid on;
title('Membrane Potentials');

subplot(4,1,4);

if M == 1
    plot(time, lif_cycle);
    hold on;
    plot(time, s);
    legend(["Pred", "True"]);
else
    plot(time, lif_cycle);
    hold on;
    plot(time, s, '--');
    legend("Pred 1","Pred 2","Pred 3","True 1","True 2","True 3");
end

grid on;
xlabel("Time");
title('Node-wise Detection');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y_tol = apply_tolerance(y_pred, s, USE_TEMP, window)

if ~USE_TEMP
    y_tol = y_pred>0;
    return;
end

y_bin = y_pred>0;
y_tol = zeros(size(y_bin));

for t=1:size(y_bin,1)
    tmin=max(1,t-window);
    tmax=min(size(y_bin,1),t+window);
    for m=1:size(y_bin,2)
        if any(y_bin(tmin:tmax,m))
            y_tol(t,m)=1;
        end
    end
end
end
