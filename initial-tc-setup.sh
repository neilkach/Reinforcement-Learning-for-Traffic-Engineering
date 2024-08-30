#!/bin/bash
set -x

# The interface to use
INTERFACE=ens5

# The base port for each tunnel
TUNNEL_BASE_PORT=51000

# Number of tunnels
NUM_TUNNELS=4

# The qdisc handle for the root
QDISC_HANDLE=1

# The flow ID to start incrementing from for each tunnel's origin traffic
SRC_FLOWID_BASE=10

# The flow ID to start incrementing from for each tunnel's destination traffic
DST_FLOWID_BASE=20

# Reset by deleting the qdisc
tc qdisc del dev $INTERFACE root

# Set up the root qdisc on the interface
tc qdisc add dev $INTERFACE root handle ${QDISC_HANDLE}: htb default 30 r2q 1000
tc class add dev $INTERFACE parent ${QDISC_HANDLE}: classid ${QDISC_HANDLE}:30 htb rate 1gbit burst 15k

# Set up the filters for each tunnel
for tunnel in 0 1 2 3; do 
    port_num=$(( $TUNNEL_BASE_PORT + $tunnel ))
    src_flowid=$(( $SRC_FLOWID_BASE + $tunnel ))
    dst_flowid=$(( $DST_FLOWID_BASE + $tunnel ))
    # Create a filter for traffic originating at the tunnel
    tc class add dev $INTERFACE parent ${QDISC_HANDLE}: classid ${QDISC_HANDLE}:${src_flowid} htb rate 1gbit burst 15k
    tc filter add dev $INTERFACE protocol ip parent ${QDISC_HANDLE}: prio 3 u32 match ip sport $port_num 0xffff flowid ${QDISC_HANDLE}:${src_flowid}
    # Create a netem qdisc for the tunnel's origin traffic
    tc class add dev $INTERFACE parent ${QDISC_HANDLE}: classid ${QDISC_HANDLE}:${dst_flowid} htb rate 1gbit burst 15k
    tc qdisc add dev $INTERFACE parent ${QDISC_HANDLE}:${src_flowid} handle ${src_flowid}: netem
    # Create a filter for traffic destined for the tunnel
    tc filter add dev $INTERFACE protocol ip parent ${QDISC_HANDLE}: prio 4 u32 match ip dport $port_num 0xffff flowid ${QDISC_HANDLE}:${dst_flowid}
    # Create a netem qdisc for the tunnel's return traffic
    tc qdisc add dev $INTERFACE parent ${QDISC_HANDLE}:${dst_flowid} handle ${dst_flowid}: netem
done
