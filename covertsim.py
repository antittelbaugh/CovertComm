#!/usr/bin/python3
"""
covertsim.py:

Simple covert communication using Mininet-Optical's simulation API.

Note: This is similar to, but currently slightly different from, the
full end-to-end emulation API, and it does not use the emulated
dataplane or control plane.

Bob Lantz, March 2022

Additions made by Tyler Mills, March 2022. 

"""

from mnoptical.network import Network
# Note: Fiber() is a length of fiber and Segment() is a
# (Fiber, Amplifier) tuple
from mnoptical.link import Span as Fiber, SpanTuple as Segment
from mnoptical.node import Transceiver, Roadm, LineTerminal
from mnoptical.units import abs_to_db

import numpy as np
import matplotlib.pyplot as plt
import random as rand
import sys

# Units
km = dB = dBm = 1.0
m = .001

# ROADM port numbers (arbitrary for simulation)
LINEIN = 0
LINEOUT = 1
ADD = 1
DROP = 2

# Terminal TX/RX port numbers (arbitrary for simulation)
TX = 1
RX = 2



# _________________________________________________________________________
# FIXED Topology Parameters (variable ones are at the very bottom)

# Currently, changing these will cause problems bc ch5 for Alice is indexed explicitly as [4] for Willie OSNR
TXCOUNT = 10  # 1 + number of background channels (1..TXCOUNT)
CH5ALICE = 5  # Alice's covert channel

# _________________________________________________________________________



"""
# ADD VARIANCE TO ALICE POWER: +/- randomly up to a given % 

power_a_var = 0.01  # percent variance of Alice power
power_a = power_a + (rand.random()*power_a_var*power_a*(-1)**(round(rand.random())))*dBm
new power = old power + or - random times given percent of old power

"""



# Network topology
def createnetwork(power_a, length_bga, ab_spans, length_ab_spans, ab_span_amp_gain, willie_loc_in_ab_spans):
    """We model a simple network for covert communication.
       Background traffic originates at t0 and is multiplexed
       by r0, amplified by boost0, and received at r1's line input.

       Alice transmits her covert communication from t1, and it is
       added to the background traffic by r1.

       r0 -> boost0 -> r1 -> boost1 --> tap ("amplifier") --> r2
       |               |                |                     |
       t0 (bg)         t1 (Alice)       Willie                t2 (Bob)

       We assign the port numbers explicitly and symmetrically:

       r0 --> LINEIN:r1:LINEOUT --> LINEIN:r2
       |ADD          |ADD                  |DROP
       t0:TX...      t1:TX                 t2:RX

       Note 1: Getting the port assignment right and the rules right
       is an essential, but tricky, part of getting any SDN
       network to work properly!

       Note 2: The tap/amplifier only has a single input and output
       port so we don't have to worry about its port numbering.

       For now, we model Willie by monitoring tap's input signal.

       Background traffic is on channels 1-10 but channel 5 isn't
       passed by r1.  Alice's traffic is on channel 5 and is added at
       r1."""


    net = Network()


    def Span(length, amp='', **params):
        "Return a segment of fiber with an optional compensating amp"

        assert isinstance(amp, str), f'Span(): {amp} is not a string'

        if amp:
            amp = net.add_amplifier(amp, **params)

        return Segment(span=Fiber(length=length), amplifier=amp)



    # Background traffic generator
    transceivers = [Transceiver(i,f'tx{i}',0*dBm)
                    for i in range(1, TXCOUNT+1)]
    t0 = net.add_lt('t0', transceivers=transceivers )
    r0 = net.add_roadm('r0', monitor_mode='out' )

    # Alice & Bob's respective terminals and roadms
    # Note low transmit power for Alice (Bob's tx power isn't used)
    t1 = net.add_lt('t1', transceivers=[Transceiver(1,'tx1', power_a)],
                    monitor_mode='out')
    t2 = net.add_lt('t2', transceivers=[Transceiver(1,'tx1',0*dBm)],
                    monitor_mode='in')
    r1 = net.add_roadm('r1', monitor_mode='out')
    r2 = net.add_roadm('r2', monitor_mode='in')
    
    # Background traffic goes from r0 -> boost0 -> 25km -> r1
    boost0 = net.add_amplifier('boost0', target_gain=17*dB, boost=True,
                               monitor_mode='out')
    
    #amp0 = net.add_amplifier('amp0', target_gain=25*.2)  # From before the new Span()
    spans0 = [Span( length=length_bga, amp='amp0', target_gain=25*.2)]
    net.add_link(
        r0, r1, src_out_port=LINEOUT, dst_in_port=LINEIN,
        boost_amp=boost0, spans=spans0)


    # Merged traffic goes from r1 -> boost1 -> __km -> tap -> __km -> r2
    # VARIABLE NUMBER OF SPANS ON ALICE TO BOB LINK; DIFFERENT LOCATIONS FOR WILLIE!      *      vvvvvvvvvvvv


    """    
    # General-purpose amp for given number of amps on the Alice to Bob link
    amp_ab = net.add_amplifier('amp_ab', target_gain=ab_span_amp_gain, monitor_mode='in')

    # Same as above, but called tap, so that it represents Willie's position. 
    # (see usage below). This way we can put Willie at different amps. 
    # So he isn't a 0-gain amp or a splitter here. Maybe we'll do that later.
    tap = net.add_amplifier('tap', target_gain=ab_span_amp_gain, monitor_mode='in')
    """


    # General-purpose span for given number of spans on the Alice to Bob link
    spans_ab = [] #[Span(length=length_ab_spans, amp=amp_ab)] -- Start empty so that first can be tap, if spec'd
    
    # Spec'd number of spans on the Alice to Bob link, each ending in an amp with a spec'd gain
    for s in range(1, ab_spans + 1):

        params = dict(target_gain=ab_span_amp_gain, monitor_mode='in')

        if s != willie_loc_in_ab_spans:
            spans_ab.append(Span(length=length_ab_spans, amp=f'amp_ab_{s}', **params))
        
        # For the spec'd Willie location, put another of the same amp, but named 'tap'
        elif s == willie_loc_in_ab_spans:
            spans_ab.append(Span(length=length_ab_spans, amp='tap', 
                target_gain=ab_span_amp_gain, monitor_mode='in'))


    # End the span with a length of fiber with no amp at the end, like in the original span:
    spans_ab.append(Span(length=length_ab_spans))

    print('\n*** Current Alice to Bob Span:\n', spans_ab, '\n')

    # The single boost amp at the start of the A to B span
    boost_ab = net.add_amplifier('boost_ab', target_gain=3*dB, boost=True, monitor_mode='out')
    
    net.add_link(r1, r2, src_out_port=LINEOUT, dst_in_port=LINEIN,
        boost_amp=boost_ab, spans=spans_ab )
    
    #      *      *      *      *      *      *      *      *      *      *      *      *      ^^^^^^^^^^^^


    """

    # Original:                                 *  *  *

    tap = net.add_amplifier('tap', target_gain=0, monitor_mode='in')
    boost1 = net.add_amplifier('boost1', target_gain=3*dB, boost=True, monitor_mode='out')
    spans1 = [Span(length=length_ab_spans, amp=tap), Span(length=5*km)]

    net.add_link(
        r1, r2, src_out_port=LINEOUT, dst_in_port=LINEIN,
        boost_amp=boost1, spans=spans1 )
    """
    # ------------

    # Background traffic add links at r0
    for i in range(TXCOUNT):
        net.add_link(t0, r0, src_out_port=TX+i, dst_in_port=ADD+i,
                     spans=[Span(1*m)])

    # Local add link at r1 (Alice) and drop link at r2 (Bob)
    net.add_link(t1, r1, src_out_port=TX, dst_in_port=ADD, spans=[Span(1*m)])
    net.add_link(r2, t2, src_out_port=DROP, dst_in_port=RX, spans=[Span(1*m)])

    return net


def configroadms(net):
    "Configure ROADMs"
    r0, r1, r2 = [net.name_to_node[f'r{i}'] for i in (0, 1, 2)]

    # r0 multiplexes all background channels onto its line out
    for i in range(TXCOUNT):
        r0.install_switch_rule(
            in_port=ADD+i, out_port=LINEOUT, signal_indices=[1+i])

    # r1 passes all channels except 5
    for i in range(TXCOUNT):
        if 1+i != CH5ALICE:
            r1.install_switch_rule(
                in_port=LINEIN, out_port=LINEOUT, signal_indices=[1+i])

    # Channel 5 added at r1
    r1.install_switch_rule(
        in_port=ADD, out_port=LINEOUT, signal_indices=[CH5ALICE])

    # Channel 5 dropped at r2
    r2.install_switch_rule(
        in_port=LINEIN, out_port=DROP, signal_indices=[CH5ALICE])

    """
    # Dump ROADM connections and flow tables
    for roadm in r0, r1, r2:
        print(f'*** {roadm} connections and flow tables')
        print(f'*** {roadm} inputs: {roadm.port_to_node_in}')
        print(f'*** {roadm} outputs {roadm.port_to_node_out}')
    """


def configterminals(net):
    "Configure terminals and transceivers"
    t0, t1, t2 = [net.name_to_node[f't{i}'] for i in (0, 1, 2)]

    # Configure background transmitters
    for i in range(TXCOUNT):
        t0.assoc_tx_to_channel(
            t0.id_to_transceivers[1+i], 1+i, out_port=TX+i)

    # Configure Alice's transmitter and Bob's receivers
    t1.assoc_tx_to_channel(
        t1.id_to_transceivers[1], CH5ALICE, out_port=TX)
    t2.assoc_rx_to_channel(
        t2.id_to_transceivers[1], CH5ALICE, in_port=RX)

    # Turn on all transceivers
    t2.turn_on()
    t1.turn_on()
    t0.turn_on()


# Monitoring helper functions
def getsignalwatts(node, port=None):
    "Return monitored signal, ase noise, and nli noise power in watts"
    monitor = node.monitor # Access the monitors on any nodes that have them
    return {s.index: {'pwrW': monitor.get_power(s),
                      'aseW': monitor.get_ase_noise(s),
                      'nliW': monitor.get_nli_noise(s)}
            for s in monitor.get_optical_signals(port)}

def wtodbm(W):
    "Return watts as dBm"
    return abs_to_db(W*1000.0) if W != 0 else float('-inf')

def printdbm(sigwatts):
    "Print signal watts as dBm"
    for ch, entries in sigwatts.items():
        pdbm = wtodbm(entries['pwrW'])
        adbm = wtodbm(entries['aseW'])
        ndbm = wtodbm(entries['nliW'])
        print(f'ch{ch}: pwr {pdbm:.2f}dBm '
              f'ase {adbm:.2f}dBm nli {ndbm:.2f}dBm')
    print()


# Plot Network Graph
def plotnet(net, outfile="covertsim.png", directed=True, layout='circo',
            colorMap=None, title='Covert Communication Network'):
    "Plot network graph to outfile"
    try:
        import pygraphviz as pgv
    except:
        print(
            '*** Note: Please install python3-pygraphviz for plotting\n')
        return
    color = {Roadm: 'red', LineTerminal: 'blue'}
    if colorMap:
        color.update(colorMap)
    nfont = {'fontname': 'helvetica bold', 'penwidth': 3}
    g = pgv.AGraph(strict=False, directed=directed, layout=layout,
                   label=title, labelloc='t', **nfont)
    roadms = net.roadms
    terms = net.line_terminals
    amps = net.amplifiers
    nodes = terms + roadms
    colors = {node: color.get(type(node), 'black') for node in nodes}
    for node in nodes:
        g.add_node(node.name, color=colors[node], **nfont)
    # Only plot 1 link per pair
    linkcount = {}
    for link in net.links:
        node1, node2 = link.src_node, link.dst_node
        count = linkcount.get((node1,node2), 0)
        linkcount[node1, node2] = count + 1
        if count >= 1: continue
        port1 = node1.node_to_port_out[node2]
        port1 = port1[0] if len(port1)<2 else f'{port1[0]}...'
        port2 = node2.node_to_port_in[node1]
        port2 = port2[0] if len(port2)<2 else f'{port2[0]}...'
        label = f'{node1}:{port1}'
        
        gain_sum = 0  #                 ***
        
        for span, amp in link.spans:
            label += f' --> {span.length/1e3}km'

            # HOW GET boost amp gains for each link            ?
            ##gain_sum += span.target_gain

            if amp:
                label += f' --> {amp.name}'
                # Sum the gains of any amps to include in a newline of the label
                # gain_sum += amp.target_gain  # Not correct as is

        label += f' --> {node2}:{port2}   \n\n\n'
        
        # Idea: Label total gain on each link
        #label += 'Gain (TODO: add boosts!): ' + f'{gain_sum:.1f}' + ' dB \n\n\n'

        g.add_edge(node1.name, node2.name,
                   fontsize=14, fontname='helvetica bold',
                   label=label,
                   penwidth=2)

    print("*** Plotting network topology to", outfile)
    g.layout()
    g.draw(outfile)

# Plot Signals at node:port
def plotsignals(node, port=None):
    "Plot signals at node:port"
    try:
        # import matplotlib.pyplot as plt  # Imported above
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
    except:
        print('*** Note: Please install matplotlib for plotting\n')
        return
    name = node.name
    fig, ax = plt.subplots()
    title = f'Signal and noise power at {name}'
    title = title if port is None else title + f' port {port}'
    signals = node.monitor.get_optical_signals(port=port)
    signals = sorted(signals, key=lambda s: s.index)
    sigwatts = getsignalwatts(node, port)
    xvals = [s.frequency/1e12 for s in signals]
    labels = {'pwr': 'Signal power',
              'ase': 'ASE noise', 'nli': 'NLI noise'}
    for field in 'pwr', 'ase', 'nli':
        yvals = [wtodbm(sigwatts[s.index][field+'W']) for s in signals]
        assert len(xvals) == len(yvals)
        plt.scatter(xvals, yvals, label=labels[field])
    ax.get_xaxis().set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title(title)
    plt.ylabel('dBm')
    plt.xlabel('THz')
    plt.legend()
    plt.grid()
    fname = f'{name}.png' if port is None else f'{name}-{port}.png'
    print(f'*** Saving signal plot for {name} to {fname}...')
    plt.savefig(fname)


# Run tests ====================================================================================================
def run(update_net_plot, plot_willie_signals, plot_r2_signals, plot_t2_signals, 
    length_bga, ab_spans, length_ab_spans, ab_span_amp_gain, willie_loc_in_ab_spans):

    "This function contains the entire process of Alice creating digital twin networks and \
    finding the optimal (highest permitted by RE budget) powers for a given selection of   \
    optical modes (which for our purposes are pulses). Returns the optimal powers and \
    resulting covert bits received by Bob (and bitrate), as well as the RE at Willie -- \
    all at the given selection of optical modes (pulses)."


    #___________________________________________________________________________________________________________
    # Internal parameters chosen by Alice. We could make these function arguments like the others. 

    # Maximum Total Relative Entropy Alice wants to allow at Willie. 
    # Typical value is 0.05**2 = 0.0025, which bounds Willie's probability of error as > 0.45. 
    # Alice wants Willie's prob(error) as high as possible (so he can't decide if she's transmitting).
    nRE_budget = 0.05**2

    # How close [percentage] Alice will allow Willie's RE to be to her actual budget. Arbitrary. 
    RE_margin  = 0.05
    
    # Arbitrary initial test power
    power_a_test_init   = -92*dBm

    # Alice power step size for optimum power search. RE at higher modes much more sensitive to Alice power. 
    power_a_stepsize    = 2.4*dBm

    # Max number of twin networks Alice will create and calculate Willie's RE on. 
    max_twin_tests   = 100

    # This is the number of different mode numbers Alice will run through the test networks; 
    # the number of points in modes_span. Higher value --> smoother plots.
    modes_to_test    = 7
    
    # Number of modes Alice will test and transmit up to
    max_modes  = 50000

    # Log list of mode numbers Alice will actually test with:
    modes_span = np.logspace(1, np.log10(max_modes), num=modes_to_test)
    #___________________________________________________________________________________________________________
    
    # Initialize lists of Alice's optimal powers and num iterations to find them
    power_a_test_list = []
    power_a_test_num_list = []
    

    print("\n\n_______________ALICE DIGITAL TWIN TESTING_______________\n")
    # Alice creates an instance of the network defined above to find the highest power
    # she can without exceeding her bugeted relative entropy at Willie (plus a safety margin [%]). 
    # She calculates the max RE at Willie here at using the max number of modes. 
    # In the next section, Alice uses this power (and the same max number of modes to perform the
    # "real" transmission to Bob, and we'll see how many covert bits he can receive (vs. # of modes). 

    print(f'Alice power optimizations starting from: {power_a_test_init:.4f} dBm')

    # Round the mode numbers to nearest integer
    for n in range(0, len(modes_span)):
        modes_span[n] = round(modes_span[n])

    print("Inputted subset of modes to optimize power for: \n", modes_span)

    for n in modes_span:

        twin_test_counter   = 0  # Counter for number of test networks Alice will create for each mode number

        # Resetting for each loop across the mode numbers we're testing at:
        RE_budget_ratio  = 0  # Ratio of RE of highest number of modes to budget; updated in loop below
        power_a_test = power_a_test_init

        # If power results in an RE safely below the budget, increase power
        while (RE_budget_ratio < (1 - RE_margin)) and (twin_test_counter < max_twin_tests):

            print(f'\n --------  ITERATION {twin_test_counter + 1:.0f} for {n:.0f} modes  --------')

            if twin_test_counter > 0:  # start incrementing power after the first run
                power_a_test += power_a_stepsize
                print(f"Alice test power increased to: {power_a_test:.4f} dBm")
            
            print('*** Creating latest Alice digital twin network...')
            net = createnetwork(power_a_test, length_bga, ab_spans, length_ab_spans, 
                ab_span_amp_gain, willie_loc_in_ab_spans)
            # Re-running this creates identical network, unless we add randomness

            print('*** Configuring latest Alice digital twin network...')
            configroadms(net)
            configterminals(net)


            # Debugging: check OSNRs at one of the amp_ab amps. Which one if there are multiple??????
            # ***********************************  
            #      does this return the first, if there are a bunch with the same name??
            amp_ab = net.name_to_node['amp_ab_1']
            print('---------amp_ab OSNR list: ', amp_ab.monitor.get_list_osnr())


            # OSNR at Willie's tap for Alice's channel only
            tap = net.name_to_node['tap']
            willie_input_osnr_dB_list = tap.monitor.get_list_osnr()
            print('\n-----------willie_input_osnr_dB_list: ', willie_input_osnr_dB_list)
            willie_input_osnr_alice_db = willie_input_osnr_dB_list[4][1]

            willie_input_osnr_alice_lin = 10**(willie_input_osnr_alice_db / 10)
            #print(f"Willie OSNR [linear] for ch5 (Alice): {willie_input_osnr_alice_lin:.4f}")
            
            # Relative entropy per channel use at Willie
            RE = (1/2)*np.log( (1 + willie_input_osnr_alice_lin) - (1 + willie_input_osnr_alice_lin**-1)**-1 )
            nRE = n * RE  # Total RE is the RE in one use times the number of uses. Uses are optical modes.  
            # >>> nRE here is the RE at the CURRENT number of modes being used, not the max number overall being tested

            print(f"Current Willie RE at {n} modes: {nRE:.6f}")
            RE_budget_ratio = nRE / nRE_budget
            print(f"Current RE to budget ratio: {RE_budget_ratio:.6f}\n")

            
            twin_test_counter += 1

        power_a_test -= power_a_stepsize        # Take off last step, which caused RE to exceed margin
        power_a_test_list.append(power_a_test)  # Store the optimized power for this number of modes
        power_a_test_num_list.append(twin_test_counter)

        print("\n\nOPTIMAL POWER for this number of modes: ", power_a_test, "\nLIST: ", power_a_test_list)



# ___________________ v--  beginning of "real" network  --v ______________________

    # Alice iterates through her calculated powers at the test modes,
    # and we'll  see how the RE looks across the range of n. 


    power_a_test = 0  # Setting to zero in case anyone accidentally tries to use after the testing stage above.
                      # We use power_a now to indicate the "real" power going into the "real" network.
    nRE_list = []
    bobsbits_list = []

    for i, power_a in enumerate(power_a_test_list): # need to loop through both power and corresponing n <<<<<<<<<<

        n = modes_span[i]

        print("\n\n_______________ACTUAL NETWORK TEST WITH BOB AND WILLIE_______________\n")

        print(f"Currently testing Willie RE and Bob bits for {n:.0f} modes")
        print(f"With optimized Alice power: {power_a:.4f} dBm")

        # Now in the "real" network, using the optimal powers determined above
        # Create network
        print('\n*** Creating network')
        net = createnetwork(power_a, length_bga, ab_spans, length_ab_spans, 
            ab_span_amp_gain, willie_loc_in_ab_spans)
        # Re-running this does not create a new network, without added randomness
        
        # Configure network
        print('*** Configuring network')
        configroadms(net)
        configterminals(net)

        ##print('*** Monitoring signal and noise power\n')

        # Monitor Alice's transmit power _________________________________________ A L I C E
        print("\n\t\tA L I C E \n")

        print(f"Budgeted Total Relative Entropy at Willie: {nRE_budget:.6f}")
        print(f"After {RE_margin*100:.1f}% safety margin: {nRE_budget*(1-RE_margin):.6f}\n")

        t1 = net.name_to_node['t1']  # Looking up node by name
        p_a_monitor_dBm = getsignalwatts(t1)
        print("*** Transmit powers out of t1:")
        printdbm(p_a_monitor_dBm)
        
        power_a_Watts = 10**(power_a/10)/1000 # from dBm
        print(f"Alice power [W]: {power_a_Watts}")

        """
        # Monitor merged signals out of boost1 (bg + Alice)
        boost1 = net.name_to_node['boost1']
        print("*** Monitoring merged signals out of boost1:")
        sigwatts = getsignalwatts(boost1)
        printdbm(sigwatts)
        
        """
        
        # Monitor Willie (tap) signals ___________________________________________ W I L L I E
        print("\n\t\tW I L L I E \n")
        # Important!: Right now we are allowing Willie to observe 100%
        # of the signal; more realistically we might do a 99/1 split
        # by lowering the amp gain slightly and attenuating Willie's
        # signals appropriately.
        
        tap = net.name_to_node['tap']
        
        print("*** Input signals at tap (Willie)\n*** ***(NON-ATTENUATED!!):")
        sigwatts = getsignalwatts(tap)
        # Print all of Willie's received signals
        printdbm(sigwatts)
        
        # OSNR at Willie's tap for Alice's channel only
        willie_input_osnr_dB_list = tap.monitor.get_list_osnr()
        #print("\nwillie osnr list: \n", willie_input_osnr_dB_list)
        willie_input_osnr_alice_db = willie_input_osnr_dB_list[4][1]
        print(f"\nWillie OSNR [dB] for ch5 (Alice): {willie_input_osnr_alice_db:.4f}")  # Alice: 2nd spot in tuple in ch5. *******How do this generally?!
        
        willie_input_osnr_alice_lin = 10**(willie_input_osnr_alice_db / 10)
        print(f"Willie OSNR [linear] for ch5 (Alice): {willie_input_osnr_alice_lin:.4f}")
        # Relative entropy per channel use at Willie
        RE = (1/2)*np.log( (1 + willie_input_osnr_alice_lin) - (1 + willie_input_osnr_alice_lin**-1)**-1 )
        nRE = n * RE  # Total RE is the RE in one use times the number of uses. Uses are optical modes.  

        #print(f"\nWillie total relative entropy budget set by Alice: {nRE_budget:.8f}")
        
        # willie_prob_err = TBD -- or not TBD, because this behaves the same as relative entropy (Boulat)
        
        # Plot all signals received by Willie
        if plot_willie_signals: 
            plotsignals(tap, 0)

        print("\nAdding to list the nRE of this run: ", nRE)
        nRE_list.append(nRE)  # Collect the REs at Willie across the modes Alice found the best power for.
        
        print("To be plotted, nRE List: \n", nRE_list)



        # Monitor Bob ____________________________________________________________ B O B
        print("\n\t\tB O B \n")
        
        if plot_r2_signals:
            # Plot input signals at r2
            r2 = net.name_to_node['r2']
            plotsignals(r2, LINEIN)

        # Monitor Bob's received signal (t2 gets only Alice's channel)
        t2 = net.name_to_node['t2']
        print("*** Signals at t2 (Bob, receiving only Alice's channel):")
        sigwatts = getsignalwatts(t2, RX)
        printdbm(sigwatts)
        if plot_t2_signals:
            plotsignals(t2, RX)
        
        # Bob's signal and ASE powers in Watts, if needed 
        #print('bob power [W]: ', t2.monitor.get_dict_power() )
        #print('bob ase [W]: ', t2.monitor.get_dict_ase_noise() )
        
        
        # OSNR and covert bits at Bob
        osnr_bob_db_list = t2.monitor.get_list_osnr()  # This returns OSNR in dB, which can be negative. 
        osnr_bob_db = osnr_bob_db_list[0][1]
        print(f"Bob OSNR [dB]: {osnr_bob_db:.4f}")
        osnr_bob_lin = 10**(osnr_bob_db / 10) # convert OSNR dB to linear. Can't be <0 else log is undefined. 
        print(f"Bob OSNR [linear]: {osnr_bob_lin:.4f}")
        osnr_bob_peruse = osnr_bob_lin / n
        
        bobsbits = n/2 * np.log(1 + osnr_bob_lin)
        print("\nCovert bits at Bob for this run: ", bobsbits)

        # Store Bob covert bits for this current power
        bobsbits_list.append(bobsbits)
        print("To be plotted, Bobsbits list:\n", bobsbits_list)
    
    # ________________________________________________________________________ end of 'actual' loop
    # Still in def run()

    if update_net_plot: 
        # Create and save the topology drawing/schematic (covertsim.png)
        # Of the most recent network ("net")
        plotnet(net)
    else:
        print("\n*** *** Topology plot not up to date! (Toggled off)\n")


    print("Inputted subset of modes that powers were found for: \n", modes_span)
    print("\nAlice's optimal powers for these modes: \n", power_a_test_list)
    print("\nAnd number of search steps to find them: ", power_a_test_num_list)
    
    print('\nNumber of Alice to Bob spans+amps (ab_spans) for this run: ', ab_spans)
    print('And Willie is at amp: ', willie_loc_in_ab_spans)


    # Lists of: covert bits at bob, Willie RE, Alice Power, each vs. number of modes.
    # And also Alice's RE budget for Willie and her safety margin. 
    return modes_span, power_a_test_list, nRE_budget, RE_margin, nRE_list, bobsbits_list

    # ________________________________________________________________________ end of run() function
    # ________________________________________________________________________



def plotrunresults(modes_span, power_a_test_list, nRE_budget, RE_margin, nRE_list, bobsbits_list, label_strings):

    # Label for the legend in the plots. We can choose which of the three label strings to use.
    label = 'Simulated: ' + label_strings[0]

    # Label for the x-axis of all plots
    xlabel = 'Optical Pulses in a Transmission (All time slots filled)'

    plt.figure(1)
    plt.plot(modes_span, power_a_test_list, label=label)
    plt.plot(modes_span, np.log(1/np.sqrt(modes_span)) * (power_a_test_list[0]/np.log(1/np.sqrt(modes_span[-1]))), 'k', linestyle='--', 
        label='---(TODO)---  L * log(1/sqrt(n)) = -(L/2)log(n)')
    plottitle = 'Alice Optimal Power [dBm] vs. Channel Uses (Optical Modes)'
    plottitle += f"\nRE safety margin: {RE_margin*100:.1f}%"
    plottitle += '\n' + label_strings[1] + '     ' + label_strings[2]
    plt.title(plottitle)
    plt.ylabel("Alice Optimal Power [dBm]")
    plt.xlabel(xlabel)
    plt.xscale('log')
    plt.legend()
    plt.grid(True)


    # Plot Willie's actual REs across the modes Alice found the best powers for. And Alice's budgeted nRE. 
    # The more modes_to_test, the finer this curve. 
    plt.figure(2)
    plt.plot(modes_span, nRE_list)
    plt.ylim(0, 1.1*nRE_budget) # Let's always plot from 0 to the budget
    #plt.xscale('log')
    plt.plot(modes_span, np.ones(len(modes_span))*nRE_budget, 'b', 
        linestyle='--', label=f'Alice nRE Budget: {nRE_budget:.6}')
    plottitle = '\nRelative Entropy at Willie [bits] for a Transmission of n modes\n vs. Number of Optical Modes in Transmission'
    plottitle += f"\nRE safety margin: {RE_margin*100:.1f}%"
    plottitle += '\n' + label_strings[1] + '     ' + label_strings[2]
    plt.title(plottitle)
    plt.ylabel("Total RE at Willie [bits]")
    plt.xlabel(xlabel)
    plt.legend()  #([plot1, plot2], ['111','222'])
    plt.grid(True)

    # Plot Bob covert bits vs. channel use
    plt.figure(3)
    plt.plot(modes_span, bobsbits_list, label=label)
    plt.plot(modes_span, np.sqrt(modes_span)*(bobsbits_list[-1]/np.sqrt(modes_span[-1])), 'k', 
        linestyle='--', label='---(TODO)--- L * sqrt(n)')
    plottitle = 'Covert Bits Received by Bob vs. Number of Optical Modes in Transmission'
    plottitle += '\n' + label_strings[1] + '     ' + label_strings[2]
    plt.title(plottitle)
    plt.ylabel("Covert Bits [bits]")
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(True)

    
    # Time for time?                                        ***********************
    time_per_mode = 1e-9  # nanosecond pulses (?)
    transmit_time_ms = modes_span * time_per_mode * 1000  # Total transmission times across the mode numbers
    print(f"\n\nAlice's total transmission time for a burst with {modes_span[-1]:.0f} modes: {transmit_time_ms[-1]:.6f} [ms]")
    # Bit rate at the max number of modes for a given pulse in [bits/sec]: 
    maxmode_covert_bitrate = bobsbits_list[-1]/(transmit_time_ms[-1]/1000)  # bit/s
    print(f"Alice's covert bit rate to Bob for max # modes: {(maxmode_covert_bitrate/1000):.2f} [kbit/s]\n")
    
    bob_bitrate_list_kbit = (bobsbits_list/(transmit_time_ms/1000))/1000  # kbits/s from the second "/1000"
    plt.figure(4)
    plt.plot(modes_span, bob_bitrate_list_kbit)
    plt.plot(modes_span, (1/np.log(modes_span))*(bob_bitrate_list_kbit[-1]/(1/np.log(modes_span[-1]))), 
        'k', linestyle='--', label='---(TODO)--- L * 1/log(n)')
    plottitle = 'Covert Bit Rate to Bob [kbits/s] vs. Optical Modes in Transmission'
    plottitle += '\nPulse Duration: ' + f'{time_per_mode*10**9:.2f} ns'
    plottitle += '\n' + label_strings[1] + '     ' + label_strings[2]
    plt.title(plottitle)
    plt.ylabel("Covert Bit Rate [kbits/s]")
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(True)


    return None



# END OF FUNCTION DEFINITIONS
# ===============================================================================================================



# ===============================================================================================================
# S E T T I N G S
# ===============================================================================================================

# Toggles
update_net_plot     = 1  # Update the topology plot, covertsim.png, upon execution
plot_willie_signals = 0
plot_r2_signals     = 0
plot_t2_signals     = 0

# ________________________________________________________________________
# VARIABLE Topology Parameters! These will be fed to createnetwork() to iterate across topologies

length_bga      = 5.0*km  # ro -- r1 span; Background to Alice's ROADM
length_ab_spans = 5.0*km  # Alice's ROADM to Willie's tap

# length_wb       = 10.1*km  # Willie's tap to Bob's ROADM -- Willie's tap can be anywhere, so not using this?

ab_spans        = 4        # Number of spans of length length_ab_spans on the Alice to Bob link, each with amp
ab_span_amp_gain= 3*dBm    # Gain of the repeated amps on the A B span


# ********************************************************** Willie's location in the Alice to Bob span set 
# Given integer is the amp number in the overall span. 
# Willie's OSNR will be taken from the input to this amp. 
willie_loc_in_ab_spans = 3


# Number of ROADMs on the Alice to Bob link   # not currently used
num_roadms      = 1


#           + more!

# ===============================================================================================================
# PARAMETER CHECKS before running; abort if anything is set nonsensically

if (willie_loc_in_ab_spans > ab_spans) or (willie_loc_in_ab_spans < 0):
    sys.exit('\n\n***ABORT: Willie\'s location doesn\'t make sense (set by user)\n\n')

if (TXCOUNT != 10) or (CH5ALICE != 5):
    sys.exit('\n\n***ABORT: Setting TXCOUNT != 10 or CH5ALICE != 5 won\'t work (see comment where defined)\n\n')




# ===============================================================================================================
# M  A  I  N
# ===============================================================================================================


modes_span, power_a_test_list, nRE_budget, RE_margin, nRE_list, bobsbits_list = run(update_net_plot, 
    plot_willie_signals, plot_r2_signals, plot_t2_signals,
    length_bga, ab_spans, length_ab_spans, ab_span_amp_gain, willie_loc_in_ab_spans)


# Give the plot function strings describing the current topology-level parameters we're iterating over
# We can choose how to use the three label strings in the plot function based on what we're testing.
label_strings    = ['','','']
label_strings[0] = f'Tap at amp {willie_loc_in_ab_spans} of {ab_spans}'
label_strings[1] = f'Length of each AB span: {length_ab_spans}km'
label_strings[2] = f'AB span amp gains: {ab_span_amp_gain}dB'

plotrunresults(modes_span, power_a_test_list, nRE_budget, RE_margin, nRE_list, bobsbits_list, label_strings)


# Show all of the plots once all the results from all of the runs have been plotted
plt.show()







print('\n*** Done!\n\n')
print("*** NOTES:")
print("\t-Willie tap element isn't a splitter yet. No attenuation.")
print("\t-Willie is currently the input to one of the amps on the AB link. This amp has the same gain as the others.")
print("\t-Changing # of bg channels from 10 will break things \
    \n\t because ch5 for Alice is indexed explicitly as [4] at Willie OSNRs")


