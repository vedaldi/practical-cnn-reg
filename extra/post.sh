#!/bin/bash

# A version with standard binaries
make -f extra/Makefile bins
make -f extra/Makefile pack post post-doc

# A version with EC2 binaries
name=practical-cnn-reg-ec2 make -f extra/Makefile ec2bins
name=practical-cnn-reg-ec2 make -f extra/Makefile pack post
