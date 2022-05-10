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
from mnoptical.node import Transceiver, Roadm, LineTerminal, SignalTracing, Splitter
from mnoptical.units import abs_to_db


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import random as rand
import sys
import datetime

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
def createnetwork(power_a, length_bga, num_ab_spans, length_ab_spans, 
    ab_span_amp_gain, tap_loc, num_ab_roadms, span_ab_boostamp_gain, eta, not_eta):
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
            amp = net.add_amplifier(amp, **params, wdg_id='linear')

        return Segment(span=Fiber(length=length), amplifier=amp)



    # Background traffic generator
    transceivers = [Transceiver(i,f'tx{i}',0*dBm)
                    for i in range(1, TXCOUNT+1)]
    t0 = net.add_lt('Background', transceivers=transceivers )
    r0 = net.add_roadm('r0', monitor_mode='out' )

    # Alice & Bob's respective terminals and roadms
    # Note low transmit power for Alice (Bob's tx power isn't used)
    t1 = net.add_lt('Alice', transceivers=[Transceiver(1,'tx1', power_a)],
                    monitor_mode='out')
    t2 = net.add_lt('Bob', transceivers=[Transceiver(1,'tx1',0*dBm)],
                    monitor_mode='in')

    t3 = net.add_lt('Willie', transceivers=[Transceiver(1,'tx1',0*dBm)],
                    monitor_mode='in')

    r1 = net.add_roadm('r1', monitor_mode='out')
    r2 = net.add_roadm('r2', monitor_mode='in')
    
    # Background traffic goes from r0 -> boost0 -> 25km -> r1
    boost0 = net.add_amplifier('boost0', target_gain=17*dB, boost=True,
                               monitor_mode='out', wdg_id='linear')
    
    #amp0 = net.add_amplifier('amp0', target_gain=25*.2)  # From before the new Span()
    spans0 = [Span( length=length_bga, amp='amp0', target_gain=25*.2)]
    net.add_link(
        r0, r1, src_out_port=LINEOUT, dst_in_port=LINEIN,
        boost_amp=boost0, spans=spans0)


    # Merged traffic goes from r1 -> boost1 -> __km -> tap -> __km -> r2
    
    """    
    # General-purpose amp for given number of amps on the Alice to Bob link
    amp_ab = net.add_amplifier('amp_ab', target_gain=ab_span_amp_gain, monitor_mode='in')

    # Same as above, but called tap, so that it represents Willie's position. 
    # (see usage below). This way we can put Willie at different amps. 
    # So he isn't a 0-gain amp or a splitter here. Maybe we'll do that later.
    tap = net.add_amplifier('tap', target_gain=ab_span_amp_gain, monitor_mode='in')
    """

    # ================ OPTIONALLY ADDING ROADMs ON THE ALICE TO BOB LINE ==========================================
    r_ab = []
    #     list of added ROADMs on the AB line.................?
    for r in range(1, num_ab_roadms + 1):
        r_ab.append(net.add_roadm(f'r_ab_{r}', monitor_mode='in'))
        # So now r_ab_x is r_ab[x]

    # ================ OPTIONALLY ADDING AMPS 
    # ================ ON ***EACH and EVERY SPAN BETWEEN ROADMS*** ON THE ALICE TO BOB LINE =======================

    # Shorter names for these, because we have to call this function every time we link() things
    n, l, g = num_ab_spans, length_ab_spans, ab_span_amp_gain

    def spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p): 
        "Create spec'd number of spans between EACH ROADM on the Alice to Bob link.\
        Each span ends in an amp with a spec'd gain.\
        This needs to be a function because we can't reuse a single span variable over and over!"

        # General-purpose span for given number of spans on the Alice to Bob link
        spans_ab = [] #[Span(length=length_ab_spans, amp=amp_ab)] -- Start empty so that first can be tap, if spec'd
        

        for s in range(1, num_ab_spans + 1):

            params = dict(target_gain=ab_span_amp_gain, monitor_mode='in')
            spans_ab.append(Span(length=length_ab_spans, amp=f'amp_ab_{p}_{s}', **params))

            """ *** Taking this out, now that we have ROADMs and a splitter for the tap!
            if s != tap_loc:
                spans_ab.append(Span(length=length_ab_spans, amp=f'amp_ab_{s}', **params))
            
            # For the spec'd Willie location, put another of the same amp, but named 'tap'
            elif s == tap_loc:
                spans_ab.append(Span(length=length_ab_spans, amp='tap', 
                    target_gain=ab_span_amp_gain, monitor_mode='in'))
            """

        # End each span with a length of fiber with no amp:
        spans_ab.append(Span(length=length_ab_spans))
        # debug ***
        #print('\n*** span_ab (the span(s) between all ROADMs on the A to B line):\n', spans_ab, '\n')

        # Return a span that's unique for each call to this function

        return spans_ab



    def boost_amp_ab(gain, count):
        "Create a boost amp to go at the start of the spans between ROADMs on the A to B line"

        name = 'boost_ab_' + f'{count}'
        boostamp = net.add_amplifier(name, target_gain=gain, boost=True, 
            monitor_mode='out', wdg_id='linear')
        
        return boostamp


    # ================ LINKING the added ROADMs on the A to B line ===============================
    # ================ And incorporating the splitter (Willie's tap) location


    # Percentage input power lost due to absorption at Willie's tap.
    tap_abs = 100 - eta - not_eta
    print(f'\nPercentage input power lost due to absorption at Willie\'s tap: {tap_abs:.4f}%')
    
    # Splitter element. LINEOUT is port that continues on the line. Port 2 goes to Willie (t3).
    # split is a dictionary with ports mapped to percentage power passed (eta and not_eta).
    tap = net.add_node('tap', cls=Splitter, split={LINEOUT:eta, 2:not_eta})
    
    # tap_loc = n means that tap is after the span coming from r_ab_(n+1). 
    # tap_loc = 0 means it's after the span after r1 (not r_ab_1). 

    # Counter used to give every boost amp a unique name
    b = 0

    # Counter used to give every span_ab a unique name
    p = 0

    # -- CONDITIONS FOR WHERE THE TAP GOES --

    # If tap is after r1 (at the end of the span after r1)
    if tap_loc == 0:

        b += 1
        p += 1
        # Link r1 to tap
        net.add_link(r1, tap, src_out_port=LINEOUT, dst_in_port=LINEIN, 
            boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
            spans=spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p) )
        
        b += 1
        p += 1
        # Then tap to the first added ROADM, with no span after the tap
        net.add_link(tap, r_ab[0], src_out_port=LINEOUT, dst_in_port=LINEIN, 
            boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
            spans=[Span(0.001)] )

        # Now link the rest of any added ROADMs together
        for r in range(0, num_ab_roadms-1):
            b += 1
            p += 1
            # e.g. for num_ab_roadms=5, this does 0, 1, 2, 3. We do last ROADM after this loop.
            net.add_link(r_ab[r], r_ab[r+1], src_out_port=LINEOUT, dst_in_port=LINEIN, 
                boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
                spans=spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p) )
                
        b += 1
        p += 1
        # Finally, link the last of the added ROADMs to r2 (Bob)
        net.add_link(r_ab[num_ab_roadms-1], r2, src_out_port=LINEOUT, dst_in_port=LINEIN,
            boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
            spans=spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p) )

    # If the tap isn't first, and also isn't last on the line of ROADMs (which would be right before r2)
    elif tap_loc < num_ab_roadms:
        b += 1
        p += 1
        # First link r1 to the first added ROADM (with span_ab in between)
        net.add_link(r1, r_ab[0], src_out_port=LINEOUT, dst_in_port=LINEIN, 
            boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
            spans=spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p) )

        # Now link the rest of any added ROADMs together, and the tap, wherever it's located
        for r in range(0, num_ab_roadms-1):
            # e.g. for num_ab_roadms=5, this does 0, 1, 2, 3. We do last ROADM after this loop.
            
            # If next one isn't tap, then link current and next
            if r+1 != tap_loc:
                b += 1
                p += 1
                # e.g. for tap_loc=3, tap will go after r3: link(r_ab[2], tap) and link(tap, r_ab[3])
                net.add_link(r_ab[r], r_ab[r+1], src_out_port=LINEOUT, dst_in_port=LINEIN, 
                    boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
                    spans=spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p) )
        
            # Otherwise, link the current r_ab to the tap, and then the tap to the next r_ab
            elif r+1 == tap_loc:
                b += 1
                p += 1
                net.add_link(r_ab[r], tap, src_out_port=LINEOUT, dst_in_port=LINEIN, 
                    boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
                    spans=spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p) )
                b += 1
                p += 1
                net.add_link(tap, r_ab[r+1], src_out_port=LINEOUT, dst_in_port=LINEIN, 
                    boost_amp=boost_amp_ab(span_ab_boostamp_gain, b),
                    spans=[Span(0.001)] )

        # Now the last added ROADM to r2. Tap last is handled in the next if statement.
        b += 1
        p += 1
        net.add_link(r_ab[num_ab_roadms-1], r2, src_out_port=LINEOUT, dst_in_port=LINEIN, 
            boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
            spans=spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p) )

    # Willie tap is after the last added ROADM (meaning right before r2)
    elif tap_loc == num_ab_roadms:
        b += 1
        p += 1
        # First link r1 to the first added ROADM (with span_ab in between)
        net.add_link(r1, r_ab[0], src_out_port=LINEOUT, dst_in_port=LINEIN, 
            boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
            spans=spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p) )

        for r in range(0, num_ab_roadms-1):
            b += 1
            p += 1
            # So for num=5: this does 0, 1, 2, 3 (ending on r_ab_4), and we do last ROADM (5th) after this loop
            net.add_link(r_ab[r], r_ab[r+1], src_out_port=LINEOUT, dst_in_port=LINEIN, 
                boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
                spans=spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p) )
        b += 1
        p += 1
        net.add_link(r_ab[num_ab_roadms-1], tap, src_out_port=LINEOUT, dst_in_port=LINEIN,
            boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
            spans=spans_ab(num_ab_spans, length_ab_spans, ab_span_amp_gain, p) )

        b += 1
        p += 1
        net.add_link(tap, r2, src_out_port=LINEOUT, dst_in_port=LINEIN,
            boost_amp=boost_amp_ab(span_ab_boostamp_gain, b), 
            spans=[Span(0.001)] )
                

    """
    # First link r1 to the first added ROADM (with span_ab in between)
    net.add_link(r1, r_ab[0], src_out_port=LINEOUT, dst_in_port=LINEIN, 
        boost_amp=boost_ab, spans=spans_ab )


    # Next, link each of the added ROADMs together (with span_ab in between)
    # And each LINK has its own BOOST AMP. Plus any amps in the span(s) between!
    for r in range(0, num_ab_roadms-1):
        # So for num=5, this does 0, 1, 2, 3. We do last ROADM after this loop.
        net.add_link(r_ab[r], r_ab[r+1], src_out_port=LINEOUT, dst_in_port=LINEIN, 
            boost_amp=boost_ab, spans=spans_ab )

    # Finally, link the last of the added ROADMs to r2 (Bob)
    net.add_link(r_ab[num_ab_roadms-1], r2, src_out_port=LINEOUT, dst_in_port=LINEIN,
        boost_amp=boost_ab, spans=spans_ab )
    """

    # Background traffic add links at r0
    for i in range(TXCOUNT):
        net.add_link(t0, r0, src_out_port=TX+i, dst_in_port=ADD+i, spans=[Span(1*m)])

    # Local add link at r1 (Alice) and drop link at r2 (Bob)
    net.add_link(t1, r1, src_out_port=TX, dst_in_port=ADD, spans=[Span(1*m)])
    net.add_link(r2, t2, src_out_port=DROP, dst_in_port=RX, spans=[Span(1*m)])
    # Link from Willie tap OUT2 to his terminal
    net.add_link(tap, t3, src_out_port=2, dst_in_port=RX, spans=[Span(1*m)])

    return net


def configroadms(net, num_ab_roadms):
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

    # ADD Alice's channel (5) at all the intermediate ROADMs  ...............................................??
    for r in range(1, num_ab_roadms+1):
        roadm = net.name_to_node[f'r_ab_{r}']
        for i in range(TXCOUNT):
            roadm.install_switch_rule(
                in_port=LINEIN, out_port=LINEOUT, signal_indices=[1+i])


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
    #t0, t1, t2, t3 = [net.name_to_node[f't{i}'] for i in (0, 1, 2, 3)]
    t0 = net.name_to_node['Background']
    t1 = net.name_to_node['Alice']
    t2 = net.name_to_node['Bob']
    t3 = net.name_to_node['Willie']

    # Configure background transmitters
    for i in range(TXCOUNT):
        t0.assoc_tx_to_channel(
            t0.id_to_transceivers[1+i], 1+i, out_port=TX+i)

    # Configure Alice's transmitter and Bob's receivers
    t1.assoc_tx_to_channel(
        t1.id_to_transceivers[1], CH5ALICE, out_port=TX)
    t2.assoc_rx_to_channel(
        t2.id_to_transceivers[1], CH5ALICE, in_port=RX)
    t3.assoc_rx_to_channel(
        t3.id_to_transceivers[1], CH5ALICE, in_port=RX)

    # Turn on all transceivers
    t3.turn_on()
    t2.turn_on()
    t1.turn_on()
    t0.turn_on()


# Monitoring helper functions
def getsignalwatts(node, port=None):
    "Return monitored signal, ase noise, and nli noise power in Watts"
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
            colorMap=None, title='Title?'):
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
    nfont = {'fontname': 'helvetica', 'penwidth': 3, 'fontsize':100}

    title = 'Optical Covert Communication Network'
    title += '\nPlot Updated: ' + str(datetime.datetime.now())

    g = pgv.AGraph(strict=False, directed=directed, layout=layout,
                   label=title, labelloc='t', **nfont)
    roadms = net.roadms
    terms = net.line_terminals
    amps = net.amplifiers
    nodes = terms + roadms
    colors = {node: color.get(type(node), 'black') for node in nodes}
    for node in nodes:
        g.add_node(node.name, color=colors[node], **nfont)

    g.add_node('tap', color='black', **nfont)

    # Only plot 1 link per pair
    linkcount = {}
    for link in net.links:
        node1, node2 = link.src_node, link.dst_node
        count = linkcount.get((node1,node2), 0)
        linkcount[node1, node2] = count + 1
        if count >= 1: continue
        port1 = node1.node_to_port_out[node2]
        port1 = port1[0] if len(port1)>0 else f'{port1[0]}...' # edited from <2
        port2 = node2.node_to_port_in[node1]
        port2 = port2[0] if len(port2)>0 else f'{port2[0]}...'
        label = f'{node1}:{port1}'
        
        # Idea: Label total gain on each link
        # First, add boost amp gain, or 0 if there isn't a boost on the current link
        gain_sum = getattr(link.boost_amp, 'target_gain', 0)
        
        for span, amp in link.spans:
            label += f' --> {span.length/1e3}km'
            if amp:
                label += f' --> {amp.name}'
                # Then add gains of any amps in the spans of this link
                gain_sum += amp.target_gain

        label += f' --> {node2}:{port2}   \n'
        
        # Add gain sum to topology plot label for this link
        label += 'Total gain: ' + f'{gain_sum:.1f}' + ' dB\n\n\n'

        g.add_edge(node1.name, node2.name,
                   fontsize=100, fontname='helvetica',
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
    length_bga, num_ab_spans, length_ab_spans, ab_span_amp_gain, tap_loc, 
    num_ab_roadms, span_ab_boostamp_gain, eta, not_eta):

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
    power_a_test_init   = -110*dBm

    # Alice power step size for optimum power search. RE at higher modes much more sensitive to Alice power. 
    power_a_stepsize    = 0.20*dBm

    # Max number of twin networks Alice will create and calculate Willie's RE on. 
    max_twin_tests   = 400

    # This is the number of different mode numbers Alice will run through the test networks; 
    # the number of points in modes_span. Higher value --> smoother plots.
    # Total runtime is roughly *multiplied* by this value, depending on number of Willie locations, etc.
    modes_to_test    = 20
    
    # Number of modes Alice will test and transmit up to
    max_modes  = 100000

    # List of a subset of values from the range of values of n. These are the n values that everything will be tested using.
    ###modes_span = np.logspace(0, np.log10(max_modes), num=modes_to_test)
    modes_span = np.linspace(1, max_modes, num=modes_to_test)

    #___________________________________________________________________________________________________________
    
    # Initialize lists of Alice's optimal powers and num iterations to find them
    power_a_test_list_db = []
    power_a_test_num_list = []
    

    print("\n\n_______________ALICE DIGITAL TWIN TESTING_______________\n")
    # Alice creates an instance of the network defined above to find the highest power
    # she can without exceeding her bugeted relative entropy at Willie (plus a safety margin [%]). 
    # In the next section, Alice uses these powers in the
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
            net = createnetwork(power_a_test, length_bga, num_ab_spans, length_ab_spans, 
                ab_span_amp_gain, tap_loc, num_ab_roadms, span_ab_boostamp_gain, eta, not_eta)
            # Re-running this creates identical network, unless we add randomness
        
            # *****************************************************************************/////////////////
            #plotnet(net)  Don't leave this here normally; everything will lag

            print('*** Configuring latest Alice digital twin network...')
            configroadms(net, num_ab_roadms)
            configterminals(net)

            #    * * * 
            path_ch5 = SignalTracing.channel_paths(
                node=net.name_to_node['Alice'], channel=CH5ALICE)  #, channel=CH5ALICE)
            print('\n *** *** *** PATH for CH5ALICE from t1: ', path_ch5, '\n\n')

            # Debugging: check OSNRs at an amp_ab amp. Tap replaces one, so key error if you try to check that one.
            #amp_ab = net.name_to_node['amp_ab_1']
            #print('---------amp_ab OSNR list: ', amp_ab.monitor.get_list_osnr())

            # OSNR at Willie's tap (the terminal connected to it!) for Alice's channel only
            willie_terminal = net.name_to_node['Willie']
            willie_input_osnr_dB_list = willie_terminal.monitor.get_list_osnr()
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
        power_a_test_list_db.append(power_a_test)  # Store the optimized power for this number of modes
        power_a_test_num_list.append(twin_test_counter)

        print("\n\nOPTIMAL POWER for this number of modes: ", power_a_test, "\nLIST: ", power_a_test_list_db)



# ___________________ v--  beginning of "real" network  --v ______________________

    # Alice iterates through her calculated powers at the test modes,
    # and we'll  see how the RE looks across the range of n. 


    power_a_test = 0  # Setting to zero in case anyone accidentally tries to use after the testing stage above.
                      # We use power_a now to indicate the "real" power going into the "real" network.
    nRE_list = []
    bobsbits_list = []

    for i, power_a in enumerate(power_a_test_list_db): # need to loop through both power and corresponing n <<<<<<<<<<

        n = modes_span[i]

        print("\n\n_______________ACTUAL NETWORK TEST WITH BOB AND WILLIE_______________\n")

        print(f"Currently testing Willie RE and Bob bits for {n:.0f} modes")
        print(f"With optimized Alice power: {power_a:.4f} dBm")

        # Now in the "real" network, using the optimal powers determined above
        # Create network
        print('\n*** Creating network')
        net = createnetwork(power_a, length_bga, num_ab_spans, length_ab_spans, 
            ab_span_amp_gain, tap_loc, num_ab_roadms, span_ab_boostamp_gain, eta, not_eta)
        # Re-running this does not create a new network, without added randomness

        # Configure network
        print('*** Configuring network')
        configroadms(net, num_ab_roadms)
        configterminals(net)

        ##print('*** Monitoring signal and noise power\n')
        # OSNR at Willie's tap (the terminal connected to it!) for Alice's channel only
        willie_terminal = net.name_to_node['Willie']
        willie_input_osnr_dB_list = willie_terminal.monitor.get_list_osnr()
        print('\n\n-----------willie_input_osnr_dB_list: ', willie_input_osnr_dB_list, '\n\n\n')
        willie_input_osnr_alice_db = willie_input_osnr_dB_list[4][1]



        # Monitor Alice's transmit power _________________________________________ A L I C E
        print("\n\t\tA L I C E \n")

        print(f"Budgeted Total Relative Entropy at Willie: {nRE_budget:.6f}")
        print(f"After {RE_margin*100:.1f}% safety margin: {nRE_budget*(1-RE_margin):.6f}\n")

        t1 = net.name_to_node['Alice']  # Looking up node by name
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
        
        # Again, just in case
        willie_terminal = net.name_to_node['Willie']
        
        print("*** Input signals at Willie terminal from tap\n*** ***(NON-ATTENUATED!!):")
        sigwatts = getsignalwatts(willie_terminal)
        # Print all of Willie's received signals
        printdbm(sigwatts)
        
        # OSNR at Willie's tap for Alice's channel only (input at his terminal, linked to tap)
        willie_input_osnr_dB_list = willie_terminal.monitor.get_list_osnr()
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
            plotsignals(willie_terminal, 0)

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
        t2 = net.name_to_node['Bob']
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
        
        bobsbits = n/2 * np.log2(1 + osnr_bob_lin)  # Standard capacity formula for AWGN channel
        print("\nCovert bits at Bob for this run: ", bobsbits)

        # Store Bob covert bits for this current power
        bobsbits_list.append(bobsbits)
        print("To be plotted, Bobsbits list:\n", bobsbits_list)
    
    # ________________________________________________________________________ end of 'actual' loop
    # Still in def run()

    print("Inputted subset of modes that powers were found for: \n", modes_span)
    print("\nAlice's optimal powers [dBm] for these modes: \n", power_a_test_list_db)
    print("\nAnd number of search steps to find them: ", power_a_test_num_list)
    
    print('\nNumber of Alice to Bob spans+amps (num_ab_spans) for this run: ', num_ab_spans)
    #print('And Willie is at amp: ', tap_loc)


    # Lists of: covert bits at bob, Willie RE, Alice Power, each vs. number of modes.
    # And also Alice's RE budget for Willie and her safety margin. 
    return modes_span, power_a_test_list_db, nRE_budget, RE_margin, nRE_list, bobsbits_list, net

    # ________________________________________________________________________ end of run() function
    # ________________________________________________________________________



def plotrunresults(modes_span, power_a_test_list_db, nRE_budget, RE_margin, 
    nRE_list, bobsbits_list, label_strings, time_per_use):
    "Plots the result data (vs. number of pulses) and the theoretical fits.\
    Plots are finished in the other function."

    # Label for the legend in the plots. We can choose which of the three label strings to use.
    label = 'Simulated: ' + label_strings[0]

    # Label for the x-axis of all plots
    xlabel = 'n, Channel Uses in a Transmission (All slots filled)'


    # Alice's powers
    plt.figure(1)
    plt.plot(modes_span, power_a_test_list_db, label=label)
    
    # We calculate c_cov in Watts (Alice's powers are converted dBm --> W in calc_c_cov())
    c_cov = calc_c_cov(modes_span, power_a_test_list_db)
    # Now we scale the expected power dropoff, 1/sqrt(n) by c_cov, THEN convert that WITH n to log scale:
    power_a_empirical = 10*np.log10(1000*c_cov/np.sqrt(modes_span))
    plt.plot(modes_span, power_a_empirical, linestyle='--', 
        label=f'Empirical: c_cov/np.sqrt(n), c_cov = {10*np.log10(1000*c_cov):.2f}dBm')
    #plt.plot(modes_span, c_cov*np.log(1/np.sqrt(modes_span)), 'm')

    # Willie's actual nRE across the modes Alice found the best powers for
    plt.figure(2)
    plt.plot(modes_span, nRE_list, label=label)
    #plt.ylim(0.8*min(nRE_list), 1.1*nRE_budget)
    plt.ylim(0.0015, 0.0030)
    # 4/21/22: No theoretical curve for RE; it stays constant.

    # Bob's covert bits    
    # Get covert capacity, L, to scale empirical curve for Bob's covert bits
    L = calc_L(modes_span, bobsbits_list)

    plt.figure(3)
    plt.plot(modes_span, bobsbits_list, label=label)
    plt.plot(modes_span, L*np.sqrt(modes_span), linestyle='--',
        label=f'Empirical: L * sqrt(n), L = {L:.6f}')
    
    # Alice covert bitrate to Bob [s]
    transmit_times = modes_span * time_per_use  # Total transmission times across the mode numbers
    print(f"\n\nAlice total transmission time with {modes_span[-1]:.0f} modes: {transmit_times[-1]*10**6:.6f} [us]")
    # Bit rate at the max number of modes for a given pulse in [bits/sec]: 
    #maxmode_covert_bitrate = bobsbits_list[-1]/(transmit_times[-1])  # bit/s
    #print(f"Alice's covert bit rate to Bob for max # modes: {(maxmode_covert_bitrate/1000):.2f} [kbit/s]\n")
    
    bob_bitrate_list_kbit = (bobsbits_list/(transmit_times))/1000  # kbits/s from the second "/1000"
    plt.figure(4)
    plt.plot(modes_span, bob_bitrate_list_kbit, label=label)
    # Empirical: same as for bits just divided by transmit times (and 1/1000 for kbit)
    plt.plot(modes_span, L*np.sqrt(modes_span)/(1000*transmit_times), linestyle='--',
       label=f'Empirical: L*sqrt(n)/(T*n), L = {L:.6f}')
    plt.yscale('log')
    #plt.ylim(0, 2*bob_bitrate_list_kbit[2])  # Feel free to change, as needed

    return None



def finishplots(modes_span, nRE_budget, RE_margin, label_strings, time_per_use, bobsbits_list):
    "Complete the 4 plots in plotrunresults(), adding labels, titles, and the empirical curves."

    # Label for the x-axis of all plots
    xlabel = 'n, Channel Uses in a Transmission (All slots filled)'
    # Label for the part of the plot titles after the dependent variable (so it's easy to change all of them)
    plottitle_vs = ' vs. Optical Pulses in a Transmission (All slots filled)'

    plt.figure(1)
    plottitle = 'Alice Optimal Power [dBm]'
    plottitle += plottitle_vs
    plottitle += f"\nRE safety margin: {RE_margin*100:.1f}%"
    plottitle += '\n' + label_strings[1] + '  |  ' + label_strings[2]
    plt.title(plottitle)
    plt.ylabel("P, Alice Optimal Power [dBm]")
    plt.xlabel(xlabel)
    plt.xscale('log')
    plt.legend()
    plt.grid(True)

    # Willie nRE
    plt.figure(2)
    # Alice's budget for the nRE (constant)
    plt.plot(modes_span, np.ones(len(modes_span))*nRE_budget, 'b', 
        linestyle='--', label=f'Alice nRE Budget: {nRE_budget:.4}')
    plt.plot(modes_span, np.ones(len(modes_span))*nRE_budget*(1-RE_margin) , 'b', 
        linestyle=':', label=f'Alice nRE Budget - Margin: {(1- RE_margin)*nRE_budget:.4}')
    plottitle = '\nRelative Entropy at Willie [bits]'
    plottitle += plottitle_vs
    plottitle += f"\nRE safety margin: {RE_margin*100:.1f}%"
    plottitle += '\n' + label_strings[1] + '  |  ' + label_strings[2]
    plt.title(plottitle)
    plt.ylabel("Total RE at Willie [bits]")
    plt.xlabel(xlabel)
    #plt.xscale('log')
    plt.legend()
    plt.grid(True)

    plt.figure(3)
    plottitle = 'Covert Bits Received by Bob'
    plottitle += plottitle_vs
    plottitle += '\n' + label_strings[1] + '  |  ' + label_strings[2]
    plt.title(plottitle)
    plt.ylabel("B, Covert Bits at Bob [bits]")
    #plt.yscale('log')
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(True)


    plt.figure(4)
    plottitle = 'Alice to Bob Covert Bit Rate During Transmission [kbits/s]'
    plottitle += plottitle_vs
    plottitle += '\nChannel Use Duration, T: ' + f'{time_per_use*10**9:.2f} ns'
    plottitle += '\n' + label_strings[1] + '  |  ' + label_strings[2]
    plt.title(plottitle)
    plt.xscale('log')
    plt.ylabel("Alice to Bob Covert Bit Rate During Transmission [kbits/s]")
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(True)

    # Show all the plots now that they're done
    plt.show()

    return None


def surfaceplot_vs_tap_loc(tap_loc_start, tap_loc_end, num_ab_roadms, modes_span, zlabel, zdata):
    "Make a surface plot of any of the results (lists vs. # pulses) against Willie locations"

    # Surface plot for comparison across configurations
    plt.figure(5)
    plottitle = zlabel + ' vs. Optical Pulses in a Transmission (All slots filled)'
    plottitle += ' vs. Willie Tap Location in Alice-Bob Span'
    plottitle += '\n' + label_strings[1] + '  |  ' + label_strings[2]
    plt.title(plottitle)
    ax = plt.axes(projection='3d')
    willie_locs_range = range(tap_loc_start, tap_loc_end+1)
    pulses, willie_locs = np.meshgrid(modes_span, list(willie_locs_range))
    surf1 = ax.plot_surface(pulses, willie_locs, np.array(zdata), cmap=cm.coolwarm)
    plt.xlabel('n, Channel Uses in a Transmission (All slots filled)')
    plt.ylabel('Willie Tap Position')
    plt.yticks(np.arange(1, num_ab_roadms + 1, 1.0))
    ax.set_zlabel(zlabel)
    # plt.colorbar(surf1, shrink=0.5)
    plt.show()

    return None


def calc_L(n, B):
    "Inputs should always be modes and Bob's bits. \
    Calculate the covert capacity, L, used to scale empirical curves in plots,\
    by least-squares fitting to inputted Bob's bits, B. \
    B = L*sqrt(n) --> log(B) = log(L) + (1/2)log(n) "

    # Take natural logs (or any base) of inputs
    logn = np.log(n)
    logB = np.log(B)


    # Design matrix
    X = np.vstack((np.ones(len(logn)), logn)).T

    # Simple linear regression to log(n)
    LSparams = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(logB)
    # Also works and gives same result: np.polyfit(np.log(n), B, 1)

    intercept = LSparams[0]

    L = np.exp(intercept)

    print(f'\nMost recent covert capacity, L: {L:.6f}\n bits')

    return L



def calc_c_cov(n, P_dBm):
    "Inputs should always be modes and Alice's power. \
    Calculate the coefficient, c_cov, used to scale empirical curves in plots,\
    by least-squares fitting to inputted Alice powers, P. \
    P =?? c_cov/sqrt(n) --> log(P) = log(c_cov) - (1/2)log(n) "

    # First, convert Alice's power in dBm to linear [W]:
    P = [(10**(P_dBm[i]/10))/1000 for i in range(len(P_dBm))]
    

    # Take natural logs (or any base) of inputs
    logn = np.log(n)
    logP = np.log(P)


    # Design matrix
    X = np.vstack((np.ones(len(logn)), logn)).T

    # Simple linear regression to log(n)
    LSparams = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(logP)
    # Also works and gives same result: np.polyfit(np.log(n), B, 1)

    intercept = LSparams[0]

    c_cov = np.exp(intercept)  # now in [W] (linear scale)

    # READ THIS: c_cov is in Watts now. We return it in Watts, even though
    # the plot it's used for is in dBm. We have to convert c_cov and the 
    # modes_span to log scale TOGETHER. Not just c_cov and return it in dBm. See above, where called. 
    ## No: c_cov = 10*np.log10(c_cov*1000)

    # But we convert to dBm in this print, bc it looks nicer
    print(f'\nMost recent c_cov: {10*np.log10(1000*c_cov):.4f} dBm')

    # Return in WATTS (linear scale)
    return c_cov




# END OF FUNCTION DEFINITIONS
# ===============================================================================================================

starttime = datetime.datetime.now()

# ===============================================================================================================
# S E T T I N G S
# ===============================================================================================================

# Toggles

update_net_plot     = 0  # Update the topology plot, covertsim.png, upon execution
plot_results        = 1  # Toggles the overall result plots (the whole point of this simulation)
do_surface_plot     = 1  # Surface plot of inputted data (list) vs. the different tap locations

plot_willie_signals = 0  # Plots of individual signals (we don't use much)
plot_r2_signals     = 0
plot_t2_signals     = 0

# ________________________________________________________________________
# VARIABLE Topology Parameters! These will be fed to createnetwork() to iterate across topologies

length_bga      = 5.0*km  # ro -- r1 span; Background to Alice's ROADM
length_ab_spans = 25.0*km  # Length of EACH span between intermediate ROADMs on the AB line

# length_wb       = 10.1*km  # Willie's tap to Bob's ROADM -- Willie's tap can be anywhere, so not using this?

num_ab_spans        = 2        # Number of spans of length length_ab_spans on the Alice to Bob link, each with amp
                           # Significant impact on runtime. With ~10 mode numbers, maybe 3s times this value.
print('num_ab_spans: ', num_ab_spans)

ab_span_amp_gain= 3*dBm    # Gain of the repeated amps on the AB span
print('ab_span_amp_gain: ', ab_span_amp_gain, 'dB')

span_ab_boostamp_gain = 3*dB  # Gain on the initial boost amp on the AB spans between any ROADMs.
print('Boost amp gain (amps after each ROADM on AB line): ', span_ab_boostamp_gain, 'dB')


# ************************* Willie's location in the Alice to Bob span **********************************
# 0 (before first added ROADM) to n, the number of added intermediate ROADMs. Doesn't have to start at 0. 
# tap_loc = 1
# Start and end Willie tap locations for the whole test to be run over
tap_loc_start = 0
tap_loc_end   = 5

# Number of ROADMs on the Alice to Bob link
num_ab_roadms      = 5


# Percentage of input power that makes it through/past Willie's tap to the next ROADM
eta     = 97  # [%]
# Percentage of input power Willie collects from his tap
not_eta = 2   # [%]
# Could make these random...^


# Duration of the pulses that ALice would hypothetically send. This simulation doesn't model time.
time_per_use = 20e-9  # 20ns pulses. Arbitrary. 

#           + more!

# ===============================================================================================================
# PARAMETER CHECKS before running; abort if anything is set nonsensically

if (tap_loc_end > num_ab_roadms) or (tap_loc_start < 0):
    sys.exit('\n\n*** ABORT: Willie\'s tap location doesn\'t make sense (set by user)\n\n')

if (TXCOUNT != 10) or (CH5ALICE != 5):
    sys.exit('\n\n*** ABORT: Setting TXCOUNT != 10 or CH5ALICE != 5 won\'t work (see comment where defined)\n\n')

if (eta + not_eta) > 100:
    sys.exit('\n\n*** ABORT: Eta and (1-eta) can be < 100, but not more...\n\n')



# ===============================================================================================================
# M  A  I  N
# ===============================================================================================================


# Give the plot function strings describing the current topology-level parameters we're iterating over
# We can choose how to use the three label strings in the plot function based on what we're testing.
label_strings    = ['','','']
# label_strings[0] = f'Tap at amp {tap_loc} of {num_ab_spans}'
label_strings[1] = f'Distance between ROADMs: {length_ab_spans*(num_ab_spans+1)}km'
label_strings[2] = f'Gain between ROADMs: {span_ab_boostamp_gain + ab_span_amp_gain*num_ab_spans}dB'

all_alice_powers = []
all_willie_nREs  = []
all_bob_bits     = []

for tap_loc in range(tap_loc_start, tap_loc_end + 1):

    if tap_loc != 0:
        label_strings[0] = f'Tap after ROADM {tap_loc} of {num_ab_roadms}'
    elif tap_loc == 0:
        label_strings[0] = f'Tap before first ROADM'


    # Getting all the outputs of run(), for each of these topology-level iterations
    modes_span, power_a_test_list_db, nRE_budget, RE_margin, nRE_list, bobsbits_list, net = \
        run(update_net_plot,
            plot_willie_signals, plot_r2_signals, plot_t2_signals,
            length_bga, num_ab_spans, length_ab_spans, ab_span_amp_gain,
            tap_loc, num_ab_roadms, span_ab_boostamp_gain,
            eta, not_eta)

    if plot_results:
        # Add results of current network config to plots
        plotrunresults(modes_span, power_a_test_list_db, nRE_budget, RE_margin, 
            nRE_list, bobsbits_list, label_strings, time_per_use)

    # Store results of run
    all_alice_powers.append(power_a_test_list_db)
    all_willie_nREs.append(nRE_list)
    all_bob_bits.append(bobsbits_list)
    #print('\n\nALL ALICE POWERS: \n', all_alice_powers, '\n')


runtime = datetime.datetime.now() - starttime
print('\n\n*** Runtime (until plots displayed): ', runtime)

if plot_results:
    # Finish the plots, which now have all the results
    finishplots(modes_span, nRE_budget, RE_margin, label_strings, time_per_use, bobsbits_list)


# Surface plot(s)
if do_surface_plot:
    #zlabel = 'P, Alice Optimal Power [dBm]'
    zlabel = 'B, Covert Bits at Bob'
    surfaceplot_vs_tap_loc(tap_loc_start, tap_loc_end, num_ab_roadms,
        modes_span, zlabel, zdata=all_bob_bits)

print('\n')
if update_net_plot: 
    # Create and save the topology drawing/schematic (covertsim.png)
    # of the ** most recent ** network ("net") from the most recent call to run()
    plotnet(net)
else:
    print("*** *** Topology plot not up to date! (Toggled off)\n")


print('\n*** All Done.')

""" NOTE:
It appears that Willie OSNR at far enough away locations for his tap is so bad that
Alice's power has virtually no limit. Even with absurdly high powers, the nRE at Willie is still below budget.
This might be why some power plots change abruptly with n (especially visible in surface plot). But the power
step size also affects the smoothness of the plots significantly.'
"""
