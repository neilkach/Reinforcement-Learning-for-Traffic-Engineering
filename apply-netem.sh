#!/bin/bash
set -x

INTERFACE=ens5
QDISC_HANDLE=1
FLOW_ID=$1
shift

[ ! -n "${FLOW_ID}" ] && (echo "please specify flow ID"; exit 1)

tc qdisc change dev $INTERFACE parent ${QDISC_HANDLE}:${FLOW_ID} handle ${FLOW_ID}: netem "$@"
