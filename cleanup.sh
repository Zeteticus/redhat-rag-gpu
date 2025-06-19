#!/bin/bash

podman stop -a
podman rmi -a -f
podman volume rm -a -f
podman pod prune -f
podman network prune -f
systemctl --user stop podman
sleep 2
systemctl --user start podman

# 1. Force remove the stuck container
podman rm -f redhat-rag-gpu

# 2. Check what containers exist
podman ps -a | grep redhat-rag

# 3. Remove any other redhat-rag containers
podman rm -f $(podman ps -aq --filter "name=redhat-rag")

# 4. Create the missing GPU device
sudo modprobe nvidia-modeset
sudo nvidia-smi -pm 1  # This creates missing device files

# 5. Check devices now exist
ls -la /dev/nvidia0
