
import nest
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook

# Begin nest simulation
nest.ResetKernel()
nest.resolution = 0.025

## START Parameters

tstop = 100  # seconds

# Number of postsynaptic neurons (lower motor neurons)
nl = 2

# Number of Parrot/Upper Motor Neurons -> Presynaptic
nu = 1
#w Wight of upper motor neurons
upper_weight = 20

tstim = 20  # seconds
tdur = 50  # seconds
tfreq = 40  # Hz

PW = 0.1  # ms
Amp = 80000  # pA
spike_times = np.arange(tstim*1000,(tstim+tdur)*1000,1000/tfreq)

## END Parameters

## START NEST Config

# Poisson generated sine wave that we are going to be using for replicating movement stimulation in the brain
poisson = nest.Create('sinusoidal_poisson_generator', {
    'rate': 5,
    'amplitude': 5,
    'frequency': 1
})

upper_nrns = nest.Create("parrot_neuron", nu)
lower_nrns = nest.Create('aeif_psc_delta_clopath', nl, params={
    # Example setting params, default are same as paper
    # Membrane capacitance
    'C_m': 281,
    #Leak conductance
    'g_L': 30,
    # Resting potential
    'E_L': -70.6,
    # Slope factor
    'Delta_T': 2,
    'V_th_rest': -50.4,
    'tau_w': 144.0,
    'a': 4.0,
    'b': 80.5,
    'I_sp': 400.0,
    'tau_z': 40.0,
    'tau_V_th': 50.0,
    'V_th_max': -30.4,

    'A_LTP': 65e-6,
    'A_LTD': 21e-5,
    'theta_plus': -45.3,
    'theta_minus': -70.6,
    'tau_u_bar_minus': 13.8,
    'tau_u_bar_plus': 58.7,
})

spike_times_off = spike_times + PW
fes_times = np.zeros((spike_times.size*2+1))
print(fes_times.size)
fes_times[0] = nest.resolution
fes_times[1::2] = spike_times
fes_times[2::2] = spike_times_off
fes_curr = np.zeros(fes_times.size)
fes_curr[1::2] = Amp

FES = nest.Create("step_current_generator", {
    "amplitude_times": fes_times,
    "amplitude_values": fes_curr
})

## END NEST Config

## START NEST Connections

# Set UPN activity
nest.Connect(poisson, upper_nrns, conn_spec="all_to_all")

# Connect UPN to LMN
wr = nest.Create('weight_recorder')
nest.CopyModel("clopath_synapse", "clopath_synapse_rec", {"weight_recorder": wr})
nest.Connect(upper_nrns, lower_nrns, conn_spec={
    'rule': 'fixed_indegree', 'indegree': 5
}, syn_spec={
    "synapse_model": "clopath_synapse_rec",
    "delay": nest.random.normal(50,30),
    "weight": upper_weight
})


# Connect FES
nest.Connect(FES, lower_nrns, syn_spec={"delay": nest.random.normal(10,1), "weight": 30})

## END NEST Connections

recs = nest.Create('multimeter', nl, {'record_from': ['V_m']})
spikes = nest.Create('spike_recorder', nl)
nest.Connect(recs, lower_nrns, conn_spec="one_to_one")
nest.Connect(lower_nrns, spikes, conn_spec="one_to_one")

nest.Simulate(tstop * 1000)

Vs = [x['V_m'] for x in recs.get('events')]

plt.figure()
_=[plt.plot(x) for x in Vs]

plt.title("FES stimulation of lower motor neurons over time")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")

data = wr.get()["events"]
# What exatly are senders and what exactly are targets
srcs = data["senders"]
tgts = data["targets"]
tgts_uniq = set(tgts)
wgts = data["weights"]
tims = data["times"]

plt.figure()

t,V_m = (recs.get('events')[0]['times'],recs.get('events')[0]['V_m'])
s = spikes.get('events')[0]['times']

plt.plot(t,V_m,label='Voltage',color='#25e6f7')
yl = plt.ylim()
plt.vlines(s,yl[0],yl[1],color='k',label='Spikes',linestyle='--')

plt.title("Spike activity of lower motor neurons over time")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()

plt.show()


plt.figure()
# for target in tgts_uniq:
#     sel = tgts==target
#     plt.plot(tims[sel],wgts[sel])
# plt.show()

# Get all pairs of source -> target
pairs = [(s,t) for s,t in zip(srcs,tgts)]
# Remove duplicates
pairs_u = set(pairs)
# Store an array of the weight over time for each pair
weights_ = []

# df = pd.DataFrame([tims, pairs, wgts]).transpose()
# df.columns = ['times', 'source-target', 'weight']
# df.to_excel("40Hz (500 upper-motor neuron) data.xlsx")
# count = 0
# for s,t in pairs_u:
#     sel = (tgts==t) & (srcs==s)
#     plt.plot(tims[sel],wgts[sel])
#     count += 1

# NEW GRAPH FORMAT

# Get all pairs of source -> target
pairs = [(s,t) for s,t in zip(srcs,tgts)]
# Remove duplicates
pairs_u = set(pairs)
# Store an array of the weight over tiem for each pair
weights_ = []
for s,t in pairs_u:
    sel = (tgts==t) & (srcs==s)
    weights_.append((tims[sel],wgts[sel]))

# Each pair doesn't have the same number of changes
# to mediate that we take the average over each dt
# taking the last value if none are found

dt = 500
weights = np.full( (int((tstop*1000)/dt), len(pairs_u)), upper_weight, dtype=float )
for i,(t,w) in enumerate(weights_):
    w_ = weights[:,i]
    w0 = w_[0]
    i0 = 0
    for t_ in t:
        if t_ < i0*dt:
            continue
        elif t_ < (i0+1)*dt:
            sel = (t >= i0*dt) & (t < (i0+1)*dt)
            w_[i0] = np.mean(w[sel])
            w0 = w_[i0]
            i0 = i0+1
            i_b = i0
        else:
            i_b = int(np.floor(t_/dt))
            w_[i0:i_b] = w0
            i0 = i_b
    w_[i_b:] = w0
ts = np.arange(0,tstop*1000,dt)

for w in weights.T:
    plt.plot(ts,w,color='k',linewidth = 0.1)
average = np.mean(weights,axis=1)

plt.plot(ts,average,color='r')


# Create Dataframe As Kei Specified
dat=pd.DataFrame(weights)
dat.insert(0,"Time",ts)
dat.to_csv("40Hz (500 lower-motor neuron) data.csv")

# END NEW GRAPH FORMAT

plt.arrow(45000, 21.2, 25000, 0.0, color='blue', head_length = 50, head_width = 0.01, length_includes_head = True)

plt.arrow(45000, 21.2, -25000, 0.0, color='blue', head_length = 50, head_width = 0.01, length_includes_head = True)
yl1 = plt.ylim()
plt.vlines(20000, yl1[0], 21.2 ,color='blue',linestyle='--')
plt.vlines(70000, yl1[0], 21.2 ,color='blue',linestyle='--')
plt.text(41500, 21.3, "FES on", color='blue')
plt.ylim([19, 22])
plt.title("40 Hz")
plt.xlabel("Time (ms)")
plt.ylabel("Synaptic Weight")
plt.savefig('40Hz-500neuron.pdf')
plt.show()