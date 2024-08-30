#!/bin/bash

set +x

if [ $# -lt 4 ]; then
	echo "not enough arguments"
	exit 1
fi

CLIENT_B_ADDR=10.20.0.212

ip route replace ${CLIENT_B_ADDR}/32 \
	nexthop via 192.168.1.0 weight $1 \
	nexthop via 192.168.1.1 weight $2 \
	nexthop via 192.168.1.2 weight $3 \
	nexthop via 192.168.1.3 weight $4 \
