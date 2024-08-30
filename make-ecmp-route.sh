#!/bin/bash

set +x

CLIENT_B_ADDR=10.20.0.212

ip route add ${CLIENT_B_ADDR}/32 \
	nexthop via 192.168.1.0 weight 1 \
	nexthop via 192.168.1.1 weight 1 \
	nexthop via 192.168.1.2 weight 1 \
	nexthop via 192.168.1.3 weight 1
