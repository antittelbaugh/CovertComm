#!/usr/bin/python3
"""
covertsim.py:

Simple covert communication using Mininet-Optical's simulation API.

Note: This is similar to, but currently slightly different from, the
full end-to-end emulation API, and it does not use the emulated
dataplane or control plane.

Bob Lantz, March 2022
"""

from mnoptical.network import Network
# Note: Fiber() is a length of fiber and Segment() is a
# (Fiber, Amplifier) tuple
from mnoptical.link import Span as Fiber, SpanTuple as Segment
from mnoptical.node import Transceiver, Roadm, LineTerminal
from mnoptical.units import abs_to_db

# Units
km = dB = dBm = 1.0
m = .001

# ROADM port numbers (arbitrary for simulation)
LINEIN = 0
LINEOUT = 1
ADD = 100
DROP = 200

# Terminal TX/RX port numbers (arbitrary for simulation)
TX = 100
RX = 200

# Parameters
TXCOUNT = 10  # 1 + number of background channels (1..TXCOUNT)
CH5ALICE = 5  # Alice's covert channel

# Physical model API helpers
def Span(length, amp=None):
    "Return a fiber segment of length km with a compensating amp"
    return Segment(span=Fiber(length=length), amplifier=amp)

# Network topology
def createnetwork():
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

    # Background traffic generator
    transceivers = [Transceiver(i,f'tx{i}',0*dBm)
                    for i in range(1, TXCOUNT+1)]
    t0 = net.add_lt('t0', transceivers=transceivers )
    r0 = net.add_roadm('r0', monitor_mode='out' )

    # Alice & Bob's respective terminals and roadms
    # Note low transmit power for Alice (Bob's tx power isn't used)
    t1 = net.add_lt('t1', transceivers=[Transceiver(1,'tx1',-48*dBm)],
                    monitor_mode='out')
    t2 = net.add_lt('t2', transceivers=[Transceiver(1,'tx1',0*dBm)],
                    monitor_mode='in')
    r1 = net.add_roadm('r1', monitor_mode='out')
    r2 = net.add_roadm('r2', monitor_mode='in')

    # Background traffic goes from r0 -> boost0 -> 25km -> r1
    boost0 = net.add_amplifier('boost0', target_gain=17*dB, boost=True,
                               monitor_mode='out')
    amp0 = net.add_amplifier('amp0', target_gain=25*.2)
    spans0 = [Span( length=25*km, amp=amp0)]
    net.add_link(
        r0, r1, src_out_port=LINEOUT, dst_in_port=LINEIN,
        boost_amp=boost0, spans=spans0)

    # Merged traffic goes from r1 -> boost1 -> 25km -> tap -> 25km -> r2
    tap = net.add_amplifier(
        'tap', target_gain=25*.2*dB, monitor_mode='in')
    boost1 = net.add_amplifier('boost1', target_gain=3*dB, boost=True,
                               monitor_mode='out')
    spans1 = [Span(length=25*km, amp=tap), Span(length=25*km)]
    net.add_link(
        r1, r2, src_out_port=LINEOUT, dst_in_port=LINEIN,
        boost_amp=boost1, spans=spans1 )

    # Background traffic add links at r0
    for i in range(TXCOUNT):
        net.add_link(t0, r0, src_out_port=TX+i, dst_in_port=ADD+i,
                     spans=[Span(1*m)])

    # Local add link at r1 (Alice) and drop link at r2 (Bob)
    net.add_link(
        t1, r1, src_out_port=TX, dst_in_port=ADD, spans=[Span(1*m)])
    net.add_link(
        r2, t2, src_out_port=DROP, dst_in_port=RX, spans=[Span(1*m)])

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

    # Dump ROADM connections and flow tables
    for roadm in r0, r1, r2:
        print(f'*** {roadm} connections and flow tables')
        print(f'*** {roadm} inputs: {roadm.port_to_node_in}')
        print(f'*** {roadm} outputs {roadm.port_to_node_out}')


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
    monitor = node.monitor
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
        for span, amp in link.spans:
            label += f'->{span.length/1e3}km'
            if amp:
                label += f'->{amp.name}'
        label += f'->{node2}:{port2}   \n\n\n'
        g.add_edge(node1.name, node2.name,
                   fontsize=10, fontname='helvetica bold',
                   label=label,
                   penwidth=2)
    print("*** Plotting network topology to", outfile)
    g.layout()
    g.draw(outfile)

# Plot Signals at node:port
def plotsignals(node, port=None):
    "Plot signals at node:port"
    try:
        import matplotlib.pyplot as plt
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


# Run tests
def run():
    "Run test transmission(s) on our modeled network"

    # Create network
    print('*** Creating network')
    net = createnetwork()

    # Plot to file
    plotnet(net)

    # Configure network
    print('*** Configuring network')
    configroadms(net)
    configterminals(net)

    print('*** Monitoring signal and noise power\n')

    # Monitor Alice's transmit power
    t1 = net.name_to_node['t1']
    print("*** Monitoring transmit power out of t1:")
    sigwatts = getsignalwatts(t1)
    printdbm(sigwatts)

    # Monitor merged signals out of boost1 (bg + Alice)
    boost1 = net.name_to_node['boost1']
    print("*** Monitoring merged signals out of boost1:")
    sigwatts = getsignalwatts(boost1)
    printdbm(sigwatts)

    # Monitor Willie's tap signals
    # Important!: Right now we are allowing Willie to observe 100%
    # of the signal; more realistically we might do a 99/1 split
    # by lowering the amp gain slightly and attenuating Willie's
    # signals appropriately.
    tap = net.name_to_node['tap']
    print("*** Monitoring input signals at tap (NON-ATTENUATED!!):")
    sigwatts = getsignalwatts(tap)
    printdbm(sigwatts)
    plotsignals(tap, 0)

    # Plot input signals at r2
    r2 = net.name_to_node['r2']
    plotsignals(r2, LINEIN)

    # Monitor Bob's received signal
    t2 = net.name_to_node['t2']
    print("*** Monitoring incoming signal at t2:")
    sigwatts = getsignalwatts(t2, RX)
    printdbm(sigwatts)
    plotsignals(t2, RX)

# Do it!
run()
print('*** Done!')
