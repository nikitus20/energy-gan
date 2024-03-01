#!/bin/bash

# Create the logging directory in TMPDIR
mkdir -p ${TMPDIR}/tensorboard
# Copy logs to the temporary loggin directory
cp -r ./logs_wgan ${TMPDIR}/tensorboard
# Set up forwarding name and file
PORTAL_FWNAME="$(id -un | tr '[A-Z]' '[a-z]')-tensorboard"
PORTAL_FWFILE="/home/gridsan/portal-url-fw/${PORTAL_FWNAME}"
echo "Portal URL is: https://${PORTAL_FWNAME}.fn.txe1-portal.mit.edu/"

# Put the forward URL in the forwarding file
echo "http://$(hostname -s):${SLURM_STEP_RESV_PORTS}/" > $PORTAL_FWFILE

# Ensure the forwarding file has the correct permissions
chmod u+x ${PORTAL_FWFILE}

# Load an anaconda module and start tensorboard
module load anaconda/2022a
tensorboard --logdir ${TMPDIR}/tensorboard --host "$(hostname -s)" --port ${SLURM_STEP_RESV_PORTS}

# Instruction for the user to visit the provided URL
echo "TensorBoard is running. Visit the URL provided above in your browser."

