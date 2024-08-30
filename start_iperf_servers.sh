#!/bin/bash

# List of IP addresses
ip_addresses=('192.168.1.0' '192.168.1.1' '192.168.1.2' '192.168.1.3')

# Port number for iperf servers
if [ -z "$1" ]; then
    echo "Usage: $0 <port>"
    exit 1
fi

# Assign the first argument to port variable
port=$1

# Function to start iperf server
start_iperf_server() {
    local ip=$1
    local port=$2
    echo "Starting iperf server on IP: $ip, Port: $port"
    iperf3 -s -B $ip -p $port &
}

# Iterate over IP addresses and start iperf servers
for ip in "${ip_addresses[@]}"; do
    start_iperf_server $ip $port
done

echo "iperf servers started."

# Wait to keep the script running (optional)
# Comment out the following line if you want the script to exit after starting servers
wait
